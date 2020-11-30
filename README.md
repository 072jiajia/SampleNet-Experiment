# SampleNet-Experiment
# 目前因為版本問題所以是個大失敗

下載
```
git clone https://github.com/072jiajia/SampleNet-Experiment.git
```
然後用Docker

```
docker pull asafmanor/pytorch:samplenetreg_torch1.4
docker run --runtime nvidia -v $(pwd):/workspace/ -it --name {your docker's name} asafmanor/pytorch:samplenetreg_torch1.4
```


目前有兩個bug
1. 在soft_projection中的group_point會呼叫C函式，可是會no docker image error(???)
2. (我應該之後自己解決) 我的pretrain model要用torch 1.7.0的torch.load才能載入，但為了用pointnet2要用torch 1.4.0

