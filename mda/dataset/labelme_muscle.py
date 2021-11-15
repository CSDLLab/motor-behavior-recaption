import io
import ot
import shapely
import json
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.collections import PatchCollection
from shapely import geometry
from tqdm import tqdm, trange
from pathlib import Path
from PIL import Image, ImageDraw


class LabelMeMuscle(object):
    def __init__(self, data_path):
        self.index = 0
        self.data_path = data_path
        self.raw_data_list = [str(p) for p in sorted(Path(data_path).glob('*.json'))]
        self.num_motions = len(self.raw_data_list)
        self.num_muscles = 0
        self.database = None
        self.labels = []
        self.head_labels = {}
        self.attributes = ['length', 'width', 'angle', 'cx', 'cy', 'calcium']
        self.load_data()
        self.muscle_id_map = {i: np.arange(self.num_muscles) for i in range(self.num_motions)}        
    
    def load_data(self):
        index_list = []
        for filepath in self.raw_data_list:
            index_list.append(Path(filepath).stem)
            num_muscles = 0
            with open(filepath, 'r') as fp:
                raw_data = json.load(fp)
                met_tag = {}
                for seg in raw_data['shapes']:
                    num_muscles += 1
                    if seg['label'] not in self.head_labels:
                        self.head_labels[seg['label']] = 1
                    
                    if seg['label'] not in met_tag:
                        met_tag[seg['label']] = 1
                    else:
                        met_tag[seg['label']] += 1
                        if self.head_labels[seg['label']] < met_tag[seg['label']]:
                            self.head_labels[seg['label']] = met_tag[seg['label']]
            if self.num_muscles < num_muscles:
                self.num_muscles = num_muscles
        
        head_tuples = []
        for k in sorted(self.head_labels):
            for n in range(self.head_labels[k]):
                for i in self.attributes:
                    head_tuples.append([k, n+1, i])
        multiindex = pd.MultiIndex.from_tuples(head_tuples, names=('muscle_id', 'order', 'param'))
        data = np.full((self.num_motions, len(head_tuples)), np.nan)
        index = pd.Index(index_list, name=None)
        self.database = pd.DataFrame(data=data, index=index, columns=multiindex)
   
    def decode_img(self, msg):
        msg = base64.b64decode(msg)
        buf = io.BytesIO(msg)
        img = Image.open(buf)
        return img

    def get_iou(self, a, b):
        """Compute the IoU between ploygons

        :param a: Nx2 ndarray
        :type a: np.ndarray
        :param b: Nx2 ndarray
        :type b: np.ndarray
        """
        poly_a = geometry.Polygon(a).convex_hull
        poly_b = geometry.Polygon(b).convex_hull

        if not poly_a.intersects(poly_b):
            iou = 0
        else:
            try:
                inter_area = poly_a.intersection(poly_b).area
                union_area = poly_a.area + poly_b.area - inter_area
                if union_area == 0:
                    iou = 0
                else:
                    iou = inter_area / union_area
            except shapely.geos.TopologicalError:
                print("shapely.geos.TopologicalError occured, iou set to 0")
                iou = 0
        return iou
    
    def iou_matrix_from_shapes(self, a, b):
        """Compute the iou matrix from two motions

        :param a: NxMx2 motion, N shapes and each shape with M points
        :type a: np.ndarray
        :param b: NxMx2 motion, N shapes and each shape with M points
        :type b: np.ndarray
        """
        num_a = len(a)
        num_b = len(b)
        iou_matrix = np.zeros((num_a, num_b))
        for i in range(num_a):
            for j in range(num_b):
                iou_matrix[i, j] = self.get_iou(a[i], b[j])
        return iou_matrix
    
    def dist_matrix_from_shapes(self, ma, mb):
        """Compute the wasserstein matrix from two motions

        :param a: NxMx2 motion, N shapes and each shape with M points
        :type a: np.ndarray
        :param b: NxMx2 motion, N shapes and each shape with M points
        :type b: np.ndarray
        """
        num_a = len(ma)
        num_b = len(mb)
        dist_matrix = np.zeros((num_a, num_b))
        for i in range(num_a):
            for j in range(num_b):
                a = np.ones(len(ma[i])) / len(ma[i])
                b = np.ones(len(mb[j])) / len(mb[j])
                M = np.linalg.norm(ma[i, np.newaxis, :] - mb[j, :, np.newaxis], axis=2)
                Wd = ot.emd2(a, b, M)
                dist_matrix[i, j] = Wd
        return dist_matrix
    
    def ot_assign(self, method='euclidean'):
        """Create the id maps from arbitrary motion to initial motion.
        According to the IoU of any pair of muscle pieces, 
        """
        for i in trange(1, self.num_motions, desc='Indexing'):
            a = b = np.ones(self.num_muscles) / self.num_muscles
            if method == 'euclidean':
                M = self.dist_matrix_from_shapes(self.motions[i-1], self.motions[i])
                T =ot.emd(a, b, M)
            elif method == 'IoU':
                M = self.dist_matrix_from_shapes(self.motions[i-1], self.motions[i])
                T = ot.emd(a, b, np.exp(-M))
            index_map = T.argmax(axis=1)
            self.muscle_id_map[i] = index_map[self.muscle_id_map[i-1]]
    
    def count_intensity(self):
        for i, filepath in tqdm(enumerate(self.raw_data_list), desc='Ca+ Measuring'):
            data = json.load(open(filepath))
            img = np.asarray(self.decode_img(data['imageData']))
            if len(img.shape) == 3:
                img = img[:, :, 0]
            width, height = img.shape[1], img.shape[0]
            met_tag = {}
            for muscle in data['shapes']:
                polygon = [tuple(p) for p in muscle['points']]
                img_board = Image.new('L', (width, height), 0)
                ImageDraw.Draw(img_board).polygon(polygon, outline=1, fill=1)
                mask = np.array(img_board)
                intensity = (img * mask).sum() / mask.sum() / 255.0
                if muscle['label'] not in met_tag:
                    met_tag[muscle['label']] = 1
                else:
                    met_tag[muscle['label']] += 1
                self.database[muscle['label'], met_tag[muscle['label']], 'calcium'][i] = intensity
    
    def count_axis(self):
        for i, filepath in tqdm(enumerate(self.raw_data_list), desc='Shape Measuring'):
            data = json.load(open(filepath))
            met_tag = {}
            for muscle in data['shapes']:
                polygon = geometry.Polygon(muscle['points'])
                mbr_points = list(zip(*polygon.minimum_rotated_rectangle.exterior.coords.xy))

                # calculate the length of each side of the minimum bounding rectangle
                mbr_lengths = [geometry.LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]

                # get major/minor axis measurements
                minor_axis = min(mbr_lengths)
                major_axis = max(mbr_lengths)
                segments = [geometry.LineString([a, b]) for a, b in zip(mbr_points, mbr_points[1:])]
                longest_segment = max(segments, key=lambda x: x.length)
                p1, p2 = [c for c in longest_segment.coords]
                angle = np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))
                if angle > 180:
                    angle = 360 - angle
                elif angle < 0:
                    angle = 180 + angle
                cx, cy = np.mean(muscle['label'], axis=0)
                if muscle['label'] not in met_tag:
                    met_tag[muscle['label']] = 1
                else:
                    met_tag[muscle['label']] += 1
                self.database[muscle['label'], met_tag[muscle['label']], 'length'][i] = major_axis
                self.database[muscle['label'], met_tag[muscle['label']], 'width'][i] = minor_axis
                self.database[muscle['label'], met_tag[muscle['label']], 'angle'][i] = angle
                self.database[muscle['label'], met_tag[muscle['label']], 'cx'][i] = cx
                self.database[muscle['label'], met_tag[muscle['label']], 'cy'][i] = cy
        
    def visualize_tensor(self, tensor, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        patches = []
        for rect in tensor:
            affine_transform = mtransforms.Affine2D()
            affine_transform.rotate_deg_around(x=rect[3], y=rect[4], degrees=rect[2])
            xy = (rect[3] - rect[0]/2, rect[4] - rect[1]/2)
            width = rect[0]
            height = rect[1]
            patches.append(mpatches.Rectangle(xy, width, height, transform=affine_transform))
        plot_collection = PatchCollection(patches, alpha=0.4)
        colors = 0.1 * np.arange(len(patches))
        plot_collection.set_array(colors)
        ax.add_collection(plot_collection)
        xlim = [tensor[..., 3].min(), tensor[..., 3].max()]
        ylim = [tensor[..., 4].min(), tensor[..., 4].max()]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis('equal')
    
    def save_db(self, db_path, re_map=None):
        if re_map is not None:
            muscle_axis = np.asarray(self.muscles_axis)[:, [re_map[k] for k in sorted(re_map)]]
            calcium_intensities = np.asarray(self.motions_intensities)[:, [re_map[k] for k in sorted(re_map)]]
        else:
            muscle_axis = self.muscles_axis
            calcium_intensities = self.motions_intensities
        np.savez(db_path, 
                 muscle_axis=muscle_axis,
                 calcium_intensities=calcium_intensities)
    
    def save_db(self, db_path):
        self.database.to_hdf(db_path, key='muscle')
    
    def load_npz(self, db_path):
        data = np.load(db_path)
        self.muscles_axis = data['muscle_axis']
        self.motions_intensities = data['calcium_intensities']

    def __getitem__(self, index):
        # output shape: [V, C]
        return self.database.iloc[index]
            
    def __next__(self):
        if self.index < self.num_motions:
            data = self.database.iloc[self.index]
            self.index += 1
            return data
        else:
            raise StopIteration

    def __iter__(self):
        return self


class MuscleAdjacentGraph(object):
    def __init__(self, num_nodes, edges, center, k=2, strategy='spatial', max_hop=1, dilation=1) -> None:
        super(MuscleAdjacentGraph, self).__init__()
        self.num_nodes = num_nodes
        self.edges = edges
        self.center = center
        self.k = k
        self.max_hop = max_hop
        self.dilation = dilation
        self.hop_dis = self.get_hop_distance(max_hop)
        self.adj_matrix = np.where(self.hop_dis != np.inf, self.hop_dis, 0)
    
    def get_adjacency(self, stategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_nodes, self.num_nodes))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
    
    def get_hop_distance(self, max_hop=1):
        A = np.zeros((self.num_nodes, self.num_nodes))
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1
        
        # compute the hop steps
        hop_dis = np.zeros((self.num_nodes, self.num_nodes)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis
    
    @staticmethod
    def normalize_digraph(A):
        Dl = np.sum(A, 0)
        num_nodes = A.shape[0]
        Dn = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD

    @staticmethod
    def normalize_undigraph(A):
        Dl = np.sum(A, 0)
        num_nodes = A.shape[0]
        Dn = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD

    def edge_list(self):
        return np.argwhere(self.adj_matrix == 1)
