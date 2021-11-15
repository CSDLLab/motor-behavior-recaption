# Motor Dynamics Analysis

Untangling the mechanism of neural dynamics and muscle dynamics.

## Requirements

To install requirements:
```
pip install -r requirements.txt
```

## Training

Train the models in this paper, run this command:
```
python tools/train_stgcn_generator.py --cfg experiments/stgcn_generator/calcium2muscle/ventral.yaml
```

## Evaluation

To evaluate the models.

## Dataset format

```
[
    {
        'data': pd.DataFrame,
        'images': [np.ndarray, ...]
    },
    ...
]
```
