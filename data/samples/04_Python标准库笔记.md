# Python 标准库笔记

## collections
Counter 统计元素出现次数；defaultdict 访问不存在的键时返回默认值；OrderedDict 记住插入顺序。

## functools
lru_cache 做函数结果缓存，减少重复计算；partial 固定部分参数生成新函数。

## itertools
chain 把多个可迭代对象串起来；product 求笛卡尔积；permutations / combinations 求排列组合。

## datetime 与 pathlib
datetime.strptime 解析日期字符串，strftime 格式化输出；pathlib.Path 面向对象地操作路径，替代 os.path。

## 版本与包管理
pip install 安装包，pip freeze > requirements.txt 导出依赖；虚拟环境用 python -m venv venv 创建。
