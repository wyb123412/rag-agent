# 前端 JavaScript 笔记

## 闭包
闭包是函数记住其定义时作用域的能力，即使外部函数已返回。常用于封装私有变量、防抖节流。

## 事件循环
JavaScript 单线程，事件循环处理任务队列：同步代码 → 微任务(Promise) → 宏任务(setTimeout)。微任务优先级高于宏任务。

## Promise 与 async/await
Promise 表示异步操作的最终结果，有 pending / fulfilled / rejected 三种状态；async/await 是 Promise 的语法糖，让异步代码像同步一样可读。

## 深拷贝与浅拷贝
浅拷贝只复制引用（Object.assign、展开运算符）；深拷贝用 JSON.parse(JSON.stringify()) 或 structuredClone。
