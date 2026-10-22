# Linux 常用命令

## 文本处理
grep 按正则搜索文本（grep -n 显示行号、grep -i 忽略大小写）；sed 流式编辑（sed 's/旧/新/g' 替换）；awk 按列处理（awk '{print $1}' 打印第一列）。

## 文件与权限
ls -l 查看详情；chmod 755 修改权限（rwx 读写执行）；chown 改属主。

## 进程与资源
ps aux 查看进程；top 实时看 CPU/内存；kill -9 强制杀进程；df -h 看磁盘，free -h 看内存。

## 管道与重定向
管道 | 把前一个命令输出传给后一个；> 覆盖写文件，>> 追加写文件。
