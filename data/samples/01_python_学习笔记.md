# Python 学习笔记

## 装饰器
装饰器本质上是一个接收函数作为参数、并返回新函数的高阶函数。常见用途：日志、权限校验、缓存。

```python
def log(func):
    def wrapper(*args, **kwargs):
        print(f"调用 {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
```

## 生成器
生成器用 yield 关键字，惰性生成值，节省内存，适合处理大文件或无限序列。
生成器表达式形如 (x * x for x in range(10))。

## 列表推导式
列表推导式 [x for x in iterable if 条件] 比 for 循环更简洁、更快。

## 虚拟环境
python -m venv venv 创建虚拟环境，再用 pip install 安装依赖。
