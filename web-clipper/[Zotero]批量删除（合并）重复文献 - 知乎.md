# [Zotero]批量删除（合并）重复文献 - 知乎
由于重复导入或其他原因，Zotero中会出现重复文献，如果重复文献比较少的话可以点击Zotero左侧的Duplicate Items分类（文件夹），然后点击重复的文献（点击其中一条，与它重复的文献都会被选中），再点击Zotero中右上角的Merge \* items 按钮即可。

如图：

![](https://pic3.zhimg.com/v2-376c2375efd111460b4fc9ff7aa01926_b.jpg)

但如果有成百上千条重复文献，一直点鼠标也很乏味，有没有批量删除（合并）重复文献的方法呢？

可以使用JavaScript来实现。本方法见[https://forums.zotero.org/discussion/40457/merge-all-duplicates/](https://link.zhihu.com/?target=https%3A//forums.zotero.org/discussion/40457/merge-all-duplicates/)。

具体过程：

1.点击Zotero左侧的Duplicate Items分类（文件夹），按Ctrl+A，或点击Edit-Select All，全选重复文献，

2.再依次点击Tools-Developer-Run JavaScript

![](https://pic3.zhimg.com/v2-06f660b988d00ec171d7d4f6beea572a_b.jpg)

3\. 将以下代码复制到代码框中：

```js
var DupPane = Zotero.getZoteroPanes();
for(var i = 0; i < 100; i++) {
await new Promise(r => setTimeout(r, 1000));
DupPane[0].mergeSelectedItems();
Zotero_Duplicates_Pane.merge();
}

```

4.点击Run

![](https://pic2.zhimg.com/v2-ee80ee4bd76afa553a8f32d4b9165db5_b.jpg)

5.再看重复文献是不是已经被合并了。合并重复文献毕竟是删除一些文献，如果不放心可以备份一下数据（见[johnmy：\[Zotero\]数据的备份与恢复](https://zhuanlan.zhihu.com/p/350549136)），或是删除错误了，再从Trash中找回。

6.作者解释代码：

> Basically, this click 100 times the "Merge X items" button with a second waiting time in between.  
> 实际上，是点击了100次"Merge X items"，每次点击之间1s

因此，如果你的重复文献更多，可以将代码中的100改为更大的数字。