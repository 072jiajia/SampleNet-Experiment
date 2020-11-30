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


