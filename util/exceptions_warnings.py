''' Excptions and Warnings thrown by dnnb Package '''

class KernelShapeError(Exception):

    def __init__(self, x_f, k_f, s):
        s_f = '''
            Incompatible input features, kernel features and stride:
                [%s - %s] / %s + 1
                not integer
            '''
        super().__init__(s_f%(str(x_f),str(k_f),str(s)))
