# SampleNet-Experiment

## Download

```bash
git clone https://github.com/KaivinC/SampleNet-Experiment
```

## Docker Environment

```bash
docker pull asafmanor/pytorch:samplenetreg_torch1.4
docker run --runtime nvidia -v $(pwd):/workspace/ -it --name {your docker's name} asafmanor/pytorch:samplenetreg_torch1.4
```

## Building only the CUDA kernels

```bash
pip install pointnet2_ops_lib/.
```


目前有兩個bug
~~1.在soft_projection中的group_point會呼叫C函式，可是那個C函式會有no docker image error(???)~~

2.(我應該之後自己解決) 我的pretrain model要用torch 1.7.0的torch.load才能載入，但為了用pointnet2要用torch 1.4.0

3.ground truth沒跟著做sample，導致算loss時會發生問題。