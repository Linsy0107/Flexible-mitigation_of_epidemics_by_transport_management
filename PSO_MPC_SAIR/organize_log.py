import logging
import logging.config
import traceback
 
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../"))  )

'''
version - 版本，必选 int  formatters - 日志格式化器  filters - 日志过滤器  handlers - 日志处理器  loggers - 日志记录器
# # real_level of console: max(MASTER_BASIC_LEVEL, CONSOLE_BASIC_LEVEL)
# MASTER_BASIC_LEVEL = logging.INFO  # master switch: control record-level  
# CONSOLE_BASIC_LEVEL = logging.INFO  # console switch: control record-level of console  
# FILE_BASIC_LEVEL = logging.INFO  # file switch: control record-level of file 
'''

LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            'format': '%(asctime)s - %(levelname)s - %(message)s',
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "default",
        },
        "file":{
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "filename": r"./logs/log.txt",
            "formatter": "default",
            'mode': 'a',
            'encoding': 'utf-8'
        }
    },
    "loggers": {
        "console_logger": {
            "handlers": ["console"],
            "level": "ERROR",
            "propagate": False,
        },
        "file_logger":{
            "handlers": ["file"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
    "disable_existing_loggers": False,
}

class LogDecorator:
    def __init__(self, **args):
        self.args = args

    def __call__(self, func):  
        def warp(**kwargs):   
            kwargs['error:msg_code'] = kwargs.pop('msg_code')
            kwargs['error:msg'] = kwargs.pop('msg')
            # script_name = kwargs['invoker']['script_name']
            # func_name = kwargs['invoker']['func_name']
 
            # del kwargs['invoker']

            # LOGGING_CONFIG['formatters']['default']['format'] = f'%(asctime)s - {script_name} - {func_name} \n[%(levelname)s] - %(message)s'
            LOGGING_CONFIG['formatters']['default']['format'] = f'%(asctime)s - \n[%(levelname)s] - %(message)s'

            # config
            logging.config.dictConfig(LOGGING_CONFIG)

            # create a logger 
            console_logger = logging.getLogger("console_logger")
            file_logger = logging.getLogger("file_logger")

            try:
                if  'D' in kwargs['error:msg_code']:
                    console_logger.debug(kwargs)
                    file_logger.debug(kwargs) 
                elif 'I' in kwargs['error:msg_code']:
                    console_logger.info(kwargs)
                    file_logger.info(kwargs)
                elif 'W' in kwargs['error:msg_code']:
                    console_logger.warning(kwargs)
                    file_logger.warning(kwargs)
                elif 'E' in kwargs['error:msg_code']:
                    console_logger.error(kwargs)
                    file_logger.error(kwargs)
                else:
                    raise ValueError
            except ValueError as e:
                traceback.print_exc()
                return 
    
            response = func(**kwargs)

            return response
        return warp
 
@LogDecorator(args = {'a':1})  # 可以传入日志解析模块相关控制参数
def generate_log_msg(**kwargs):
    if kwargs['success'] == True:
        if kwargs['error:msg_code'] == 'I0000':
            return kwargs
        elif kwargs['error:msg_code'].startswith('D'):
            return kwargs
    elif kwargs['success'] == False:
        if  kwargs['error:msg_code'] == 'E0001':
            kwargs['error:msg'] = '变量缺失--请检查:' + str(kwargs['error:msg'])
        elif kwargs['error:msg_code'] == 'E0002':
            kwargs['error:msg'] = '数值非法--请检查:' + str(kwargs['error:msg'])
        elif kwargs['error:msg_code'] == 'E0003':
            kwargs['error:msg'] = '端口输入非法--请检查:' + str(kwargs['error:msg'])
        elif kwargs['error:msg_code'] == 'E0004':
            kwargs['error:msg'] = '数值越界--请检查:' + str(kwargs['error:msg'])
        elif kwargs['error:msg_code'] == 'E0005':
            kwargs['error:msg'] = '数据类型错误--请检查:' + str(kwargs['error:msg'])
        elif kwargs['error:msg_code'] == 'E0006':
            kwargs['error:msg'] = '发生未知错误--请检查:' + str(kwargs['error:msg'])
        elif kwargs['error:msg_code'] == 'E0007':
            kwargs['error:msg'] = '仿真异常--请检查:' + str(kwargs['error:msg'])
        elif kwargs['error:msg_code'] == 'W0001':
            kwargs['error:msg'] = '参数配置错误--请检查:' + str(kwargs['error:msg'])
        elif kwargs['error:msg_code'] == 'W0002':
            kwargs['error:msg'] = '违背约束--请检查:' + str(kwargs['error:msg'])
        elif kwargs['error:msg_code'] == 'W0003':
            kwargs['error:msg'] = '参数匹配错误--请检查:' + str(kwargs['error:msg'])
        elif kwargs['error:msg_code'] == 'W0004':
            kwargs['error:msg'] = '发生未知警告--请检查:' + str(kwargs['error:msg'])
            
        return kwargs