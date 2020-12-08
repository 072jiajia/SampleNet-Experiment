# SampleNet-Experiment

## Download

```bash
git clone https://github.com/072jiajia/SampleNet-Experiment.git
```
And put the dataset to the path below
<div align="left">
  <img src="https://github.com/072jiajia/SampleNet-Experiment/blob/main/DataPath.png"/>
</div>

## Docker Environment

```bash
docker pull asafmanor/pytorch:samplenetreg_torch1.4
docker run --runtime nvidia -v $(pwd):/workspace/ -it --name {your docker's name} asafmanor/pytorch:samplenetreg_torch1.4
```

## Building only the CUDA kernels

```bash
pip install pointnet2_ops_lib/.
```

# Pretrain
```
python3 pretrain.py
```


# Train SampleNet
```
python3 experiment.py
```



