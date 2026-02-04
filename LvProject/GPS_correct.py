from asyncio.windows_events import NULL
from tarfile import NUL

from PyQt5.QtWidgets import (QWidget, QVBoxLayout,QHBoxLayout, QTextEdit, QLineEdit, QPushButton,
                             QLabel, QFileDialog, QComboBox, QSizePolicy)
import os
# from openpyxl import load_workbook
import xlwings as xw
import threading
import time
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPalette, QColor

# event = threading.Event()
class GpsCorrect(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_file = ""  # 存储选中的文件路径
        self.last_file = ""  # 保存最后一次选择的文件
        # self._worker_thread = threading.Thread(
        #     target=self.trglight,
        #     daemon=True  # ← 主程序退出时自动杀死线程（兜底）
        # )
        # self._worker_thread.start()
        self.init_ui()

    def init_ui(self):

        layout = QVBoxLayout()#垂直排列

        self.text_display = QTextEdit()
        # self.text_display.setReadOnly(True)
        self.text_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.text_display)

        file_layout = QHBoxLayout()#水平排列

        # 指示灯（用 QLabel 模拟）
        # self.status_light = QLabel()
        # self.status_light.setFixedSize(20, 20)  # 圆形直径 20px
        # self.set_status(False)  # 初始为红色（未连接）


        #导入文件
        self.file_path_display = QLineEdit()
        self.file_path_display.setPlaceholderText("点击按钮选择定位日志...")
        # self.file_path_display.setReadOnly(True)  # 只读，防止手动
        # self.file_path_display.setMaximumHeight(30)
        # self.file_path_display.setFixedWidth(500)

        if self.last_file and os.path.isfile(self.last_file):
            self.file_path_display.setText(self.last_file)
        else:
            self.last_file = ""  # 确保无效路径被清空


        self.browse_button = QPushButton("📂 选择文件")
        self.browse_button.clicked.connect(self.open_file_dialog)
        self.browse_button.setMaximumHeight(30)
        self.browse_button.setFixedWidth(120)

        self.browse_butt = QPushButton("输出校正结果")
        self.browse_butt.clicked.connect(self.txt_exchage)
        self.browse_butt.setMaximumHeight(30)
        self.browse_butt.setFixedWidth(180)


        file_layout.addWidget(self.file_path_display)
        file_layout.addWidget(self.browse_button)
        file_layout.addWidget(self.browse_butt)
        # file_layout.addWidget(self.status_light)


        # 把水平布局整体加入主布局
        layout.addLayout(file_layout)

        layout.setStretch(0, 1)  # 让 text_display 可伸缩

        self.setLayout(layout)




    def open_file_dialog(self):
        """打开文件选择对话框"""
        options = QFileDialog.Options()
        # 不使用原生对话框（可选），保证跨平台一致
        # options |= QFileDialog.DontUseNativeDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择一个文件",  # 对话框标题
            "",  # 起始目录（空表示默认）
            "所有文件 (*);;文本文件 (*.txt);;Python 文件 (*.py)",  # 文件过滤器
            # options=QFileDialog.DontUseNativeDialog
        )
        if(not file_path or not file_path.strip()):
            return
        self.file_path = file_path
        self.file_path_display.setText(file_path)
        self.parse_file_dialog()
    def parse_file_dialog(self):

        self.keyWord = "/usbAppGNSS"  #需要过滤的关键词
        self.gpsBuffer = []
        self.gpsBufferDetail = []
        gpsBufferCell = {}

        try:
            if (".log" not in str(self.file_path)):
                self.text_display.setText("!!!!not log file!!!!")
                return
            lines = []
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    lines.append(line.strip())#读取每行截止到换行符
            for gpsline in lines :
                if(self.keyWord in gpsline):
                    self.gpsBuffer.append(gpsline)
                    self.text_display.append(gpsline)
                    time = self.get_filter_str(gpsline, '', ' I/')  # 纬度
                    part111 = gpsline.split(self.keyWord, 1)#防止日志错乱
                    status = int(self.get_filter_str(part111[1], '): ', ' '))
                    loc = self.get_filter_str(part111[1], 'lon::', ' lat::')#经度
                    locNum = 0.0
                    latNum = 0.0
                    if(loc and loc.strip()):
                        locNum = float(loc)  # 经度
                    lat = self.get_filter_str(part111[1], 'lat::', ' ')  # 纬度
                    if (lat and lat.strip()):
                        latNum = float(lat)  # 经度
                    # print("错误:", time,status,locNum,latNum)
                    self.gpsBufferDetail.append({time,status,locNum,latNum})

            return
        except Exception as e:
            print("错误:",e)
            return
        #打印

    def get_filter_str(self,text, left,right):#分割解析数据
        if text is None:
            return ""
        # 转为字符串（防数字等类型）
        s = str(text)
        # 按第一个 ':' 分割，取第一部分
        if(left==''):
            part = s.split(right, 1)
            return part[0]
        part = s.split(left, 1)
        if(len(part)>1):
            curstr = part[1]
            part = curstr.split(right, 1)
            if(len(part)>1):
                return part[0]
            else:
                return ""
        else:
            return ""
        # return s.split('：', 1)[0].strip()# strip() 去除前后空格 2)[0].strip()  # strip() 去除前后空格  1表示分隔1次 [1]表示第二部分

    def txt_exchage(self):
        try:
            content = self.text_display.toPlainText()
            # print(content)
            s = str(content)
            # # 按第一个 ':' 分割，取第一部分
            part = s.split('|')
            # part = [p.strip() for p in part if p.strip()]#去除空白
            self.relItmeP = []
            self.relItmePHex = []
            for k, p in enumerate(part):
                self.relItmeP.append(p.strip())
                self.relItmePHex.append(','.join(f'0x{b:02X}' for b in self.relItmeP[k].encode()))
                self.relItmePHex[k] = self.relItmePHex[k]+","
            self.text_display.clear()
            for k in range(0, len(self.relItmeP)-1 , 3):
                self.text_display.append(
                    self.relItmeP[k] + "|" + self.relItmeP[k + 1] + "|" + self.relItmeP[k + 2] + "|")
                self.text_display.append("###---===|")
                xxxZero = "0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,"
                aline = 16 * 5
                tmp = []
                #特殊处理B289和B275

                if int(self.DidLen[k//3])<=16:
                    wahtfck = self.DidLen[k//3]*5
                    tmpstr = self.relItmePHex[k+1]
                    if len(tmpstr)<aline and len(tmpstr)<=wahtfck:
                        tmpstr = tmpstr+xxxZero[0:(wahtfck-len(tmpstr))]
                        self.text_display.append(tmpstr)

                    self.text_display.append("")
                    tmpstr = self.relItmePHex[k + 2]
                    if len(tmpstr) < aline and len(tmpstr)<=wahtfck:
                        tmpstr = tmpstr + xxxZero[0:wahtfck - len(tmpstr)]
                        self.text_display.append(tmpstr)
                else:
                    totleLen= self.DidLen[k//3]
                    strlen = len(self.relItmePHex[k + 1])
                    strlenX = strlen//aline
                    x = 0
                    tmp.clear()
                    for x in range(strlenX):
                        tmp.append(self.relItmePHex[k+1][x*aline:(x+1)*aline])
                    tmp.append(self.relItmePHex[k+1][((strlenX)*aline):])
                    tmp[strlenX]= tmp[strlenX]+xxxZero[0:aline-(strlen-aline*strlenX)]
                    for x in range(totleLen//16):
                        if(x<len(tmp)):
                            self.text_display.append(tmp[x])
                        else:
                            self.text_display.append(xxxZero)
                    fcuk = (totleLen%16)*5
                    self.text_display.append(xxxZero[0:fcuk])

                    self.text_display.append("")#加换行

                    strlen = len(self.relItmePHex[k + 2])
                    strlenX = strlen // aline
                    x = 0
                    tmp.clear()
                    for x in range(strlenX):
                        tmp.append(self.relItmePHex[k + 2][x * aline:(x + 1) * aline])
                    tmp.append(self.relItmePHex[k + 2][((strlenX) * aline):])
                    tmp[strlenX] = tmp[strlenX] + xxxZero[0:aline - (strlen - aline * strlenX)]
                    for x in range(totleLen // 16):
                        if (x < len(tmp)):
                            self.text_display.append(tmp[x])
                        else:
                            self.text_display.append(xxxZero)
                    fcuk = (totleLen % 16) * 5
                    self.text_display.append(xxxZero[0:fcuk])
                self.text_display.append("###---===|")
                self.text_display.append("\n")#会自动添加换行
        except Exception as e:
            print("txt err:",e)
            return






