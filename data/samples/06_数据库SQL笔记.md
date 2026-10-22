# 数据库 SQL 笔记

## 查询基础
SELECT 列 FROM 表 WHERE 条件 ORDER BY 列 LIMIT n；JOIN 连接多表：INNER JOIN 只保留匹配行，LEFT JOIN 保留左表全部行。

## 索引
索引加速查询，B+ 树是常用实现。为 WHERE / ORDER BY / JOIN 列建索引能显著提速，但会拖慢写入、占用空间。

## 事务与 ACID
事务保证一组操作要么全部成功要么全部回滚。ACID = 原子性(Atomicity)、一致性(Consistency)、隔离性(Isolation)、持久性(Durability)。

## 隔离级别
读未提交、读已提交、可重复读、串行化，隔离级别越高并发越差、一致性越好。
