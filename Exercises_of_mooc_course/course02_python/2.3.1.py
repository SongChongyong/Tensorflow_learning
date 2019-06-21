#coding:utf-8
age = int(input("输入你的年龄：\n"))  #没有int则报错：'>' not supported between instances of 'str' and 'int'
if age>18:
    print ("大于18岁")
    print ("你成年里！")
else:
    print ("小于等于18岁")
    print ("你还未成年")


