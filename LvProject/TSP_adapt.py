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

event = threading.Event()
class TspAdapt(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_file = ""  # 存储选中的文件路径
        self.last_file = ""  # 保存最后一次选择的文件
        self.config_file = "D:\01_GeneralSoftware\33_PyCharm\workspace\LvProject\dist\last_file.json"  # 配置文件
        self.load_last_file()  # 启动时加载
        self._worker_thread = threading.Thread(
            target=self.trglight,
            daemon=True  # ← 主程序退出时自动杀死线程（兜底）
        )
        self._worker_thread.start()
        self.init_ui()

    def init_ui(self):

        layout = QVBoxLayout()#垂直排列

        self.text_display = QTextEdit()
        # self.text_display.setReadOnly(True)
        self.text_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.text_display)

        file_layout = QHBoxLayout()#水平排列

        # 指示灯（用 QLabel 模拟）
        self.status_light = QLabel()
        self.status_light.setFixedSize(20, 20)  # 圆形直径 20px
        self.set_status(False)  # 初始为红色（未连接）


        #导入文件
        self.file_path_display = QLineEdit()
        self.file_path_display.setPlaceholderText("点击按钮选择后台配置文件...")
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

        self.browse_butt = QPushButton("开始转换")
        self.browse_butt.clicked.connect(self.txt_exchage)
        self.browse_butt.setMaximumHeight(30)
        self.browse_butt.setFixedWidth(80)

        # 创建可编辑的 QComboBox
        self.combo = QComboBox()
        self.combo.setEditable(True)  # ← 关键：允许用户输入新值
        self.combo.addItems(["", "AHT 5G", "AY5", "AY7"])  # 初始选项
        self.combo.setMaximumHeight(30)
        self.combo.setFixedWidth(120)

        file_layout.addWidget(self.file_path_display)
        file_layout.addWidget(self.combo)
        file_layout.addWidget(self.browse_button)
        file_layout.addWidget(self.browse_butt)
        file_layout.addWidget(self.status_light)
        # file_layout.setStretch(0, 1)


        # 把水平布局整体加入主布局
        layout.addLayout(file_layout)

        layout.setStretch(0, 1)  # 让 text_display 可伸缩

        self.setLayout(layout)


    def txt_exchage(self):
        self.setEvent(2)  # 亮灯
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
        app = xw.App(visible=False)  # 后台启动 Excel
        checkSheet = 1
        self.setEvent(3)#亮灯
        if(".xls" not in str(self.file_path) or  ".xlsx" not in str(self.file_path)):
            self.text_display.setText("!!!!not excel file!!!!")
            return
        try:
            wb = app.books.open(self.file_path)
            if self.combo.currentText()=="":
                checkSheet=1
            else:
                checkSheet = 0
                if self.combo.currentText() not in [self.combo.itemText(i) for i in range(self.combo.count())]:
                    self.combo.addItems(self.combo.currentText())
                    #后续可以考虑保存本地文件永久添加
            if checkSheet==1:
                ws=wb.sheets[0]
            else:
                ws=wb.sheets[self.combo.currentText()]
            data=ws.used_range.value
            wb.close()
        except Exception as e:
            print("错误:",e)
            return
        try:
            # self.text_display.setText(data)
            self.itme = ["011B","apn1域名-011B:",   32,   "B247", "apn1类型-B247:", 1,
                         "013C","apn2域名-013C:",   32,   "B248", "apn1类型-B248:", 1,
                         "1061","apn3域名-1061:",   32,   "B28B", "apn1类型-B28B:", 1,
                         "031C","专网TSP域名-031C:", 50 ,  "011D", "专网TSP端口-011D:", 8 ,      "011C",  "专网TSP IP-011C:", 16,
                         "1014","公网TSP域名-1014:", 50 ,  "105E", "公网TSP端口-105E:", 8 ,      "105F", "公网TSP IP-105F:", 16,
                         "1052","PKI域名-1052:",    50 ,  "1053", "PKI端口-1053:", 8,
                         "B2CB","OTA域名-B2CB:",    50 ,  "B2CC", "OTA端口-B2CC:", 8,
                         "B289","埋点域名-B289:",    128,  "B275","日志域名-B275:", 100,         "B276", "日志端口-B276:", 8,
                         "011F","Ecall-011F:",      14,   "0124","Bcall-0124:", 14]
            self.itmeP = []
            self.num=0;
            self.DidLen = [0] * (len(self.itme)//3)
            itmeSta = [0]*(len(self.itme)//3)
            rows = [list(row) if isinstance(row, tuple) else row for row in data]
            self.text_display.clear()
            for i, row in enumerate(rows):
                for j, cell in enumerate(row):
                    if j+3<len(row):
                        try:
                            for k in range(0, len(self.itme), 3):
                                if ((k+1)<len(self.itme)) and (self.itme[k] in str(cell)) and (itmeSta[k//3]!=1):
                                    self.itmeP.append("🔥🔥🔥"+self.itme[k+1])
                                    self.itmeP.append(self.get_filter_str(row[j+2]))
                                    self.itmeP.append(self.get_filter_str(row[j+3]))
                                    # self.text_display.setText(self.itmeP[0]+"|"+self.itmeP[1]+"|"+self.itmeP[2]+"|")
                                    itmeSta[k//3]=1
                                    self.DidLen[self.num//3] = self.itme[k+2]
                                    self.text_display.append(self.itmeP[self.num]+"|"+self.itmeP[self.num+1]+"|"+self.itmeP[self.num+2]+"|")
                                    self.num += 3
                                    break
                        except Exception as e:
                            print("wocao: ",k, j, row[j+2])
                            return
                    else:
                        continue
        except Exception as e:
            print("data err:",e)
            return

    def get_filter_str(self,text):
        if text is None:
            return ""
        # 转为字符串（防数字等类型）
        s = str(text)
        # 按第一个 ':' 分割，取第一部分
        part = s.split('：', 1)
        if(len(part)>1):
            return part[1].strip()
        else:
            part = s.split('（', 1)
            if(len(part)>1):
                return part[0].strip()
            else:
                return part[0].strip()
        # return s.split('：', 1)[0].strip()# strip() 去除前后空格 2)[0].strip()  # strip() 去除前后空格  1表示分隔1次 [1]表示第二部分
    def load_last_file(self):
        """从 JSON 加载最后使用的文件"""
        if not os.path.exists(self.config_file):
            return
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                path = data.get("last_file", "")
                if isinstance(path, str) and os.path.isfile(path):
                    self.last_protected_file = path  # 先暂存
                    self.last_file = path
        except Exception as e:
            print("加载最后文件失败:", e)

        # if file_path and os.path.isfile(file_path):  # 确保是有效文件
        #     self.selected_file = file_path

            # 可选：自动滚动到末尾（长路径时）
            # self.file_path_display.setCursorPosition(0)

    def set_status(self, connected: bool):
        """设置指示灯状态"""
        if connected:
            color = "green"
            tooltip = "已连接到服务器"
        else:
            color = "red"
            tooltip = "未连接到服务器"

        # 使用样式表绘制圆形
        self.status_light.setStyleSheet(f"""
            background-color: {color};
            border-radius: 10px;
            border: 1px solid #aaa;
        """)
        self.status_light.setToolTip(tooltip)

    def setEvent(self, count):
        global event
        self.countLight = count
        event.set()
    def trglight(self):
        global event
        while(1):
            event.wait()
            self._light_on = 1
            while(self.countLight>0):
                self._toggle_status_light()
            event.clear()

    def _toggle_status_light(self):
        self.countLight-=1
        if self._light_on==1:
            color = "green"
            self.status_light.setStyleSheet(f"background: {color}; border-radius: 8px;")
            self._light_on = 0
        else:
            color = "red"
            self.status_light.setStyleSheet(f"background: {color}; border-radius: 8px;")
            self._light_on = 1
        time.sleep(0.5)
        if self.countLight<=0:
            color = "red"
            self.status_light.setStyleSheet(f"background: {color}; border-radius: 8px;")
            return

