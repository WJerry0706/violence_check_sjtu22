# violence_check_sjtu2024_group3_wengjieyu
这是SJTU2024年人工智能导论课大作业。组号:3;组员：翁解语，孔珺晓，毛锐。

`classfy.py`文件调试说明：

.ckpt文件由于传不下了,又不好放huggingface,就和我们添加的数据集一起传到了交大云盘中: [https://jbox.sjtu.edu.cn/v/list/self/1804178151677366278](https://jbox.sjtu.edu.cn/l/w1gOPw)

下载完`resnet50_pretrain_test-epoch=16-val_loss=0.06.ckpt`文件后,放到`train_logs/resnet50_pretrain_test/version_6/checkpoints`目录下.

将待分类的图片放置到`violence/test`目录下，然后命令行调用`classify.py`。

结果将输出在命令行，并以列表形式保存为`output\predictions_list_{current_time}.txt`和以张量形式保存为`predictions_tensor_{current_time}.pt`。


模型配置：详见`requirements.txt`。若缺失依赖，建议在conda环境下：

```pip install -r requirements.txt```



