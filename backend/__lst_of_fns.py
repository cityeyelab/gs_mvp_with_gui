def fn1(st):
    print('1' + str(st))
    print('nn??')

def fn2(st):
    print('2' + str(st))
    print('nn??')

def fn3(st):
    print('3' + str(st))

lst_of_fns = [fn1, fn2, fn3]


fisrt_fn = lst_of_fns[1]
fisrt_fn(1)
# result, _ = fisrt_fn(1)
# print(result)
# print(fisrt_fn)
# print(fisrt_fn(1), 'none?')