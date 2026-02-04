# client.py
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton,
                             QTabWidget, QLabel, QFormLayout, QMessageBox, QDesktopWidget, QMainWindow,
                             QSizePolicy)
import socket
import threading
import importlib


# class ClientApp(QWidget):
class ClientApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LQ TOOL BAR")
        # 显示布局
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)  # ← 这行至关重要！

        # 获取主屏幕尺寸
        desktop = QDesktopWidget()
        screen_rect = desktop.screenGeometry()
        # 设置窗口大小为屏幕的一半
        if 0:
            width = 1200
            height = 800
        else:
            width = screen_rect.width() // 2
            height = screen_rect.height() // 2
        self.resize(width, height)

        # if 0:
        #     self.tabs.setFixedSize(width, height)  #标签大小，不可调整
        # else:
        #     self.tabs.resize(width, height)  # 设置标签大小

        ######----------增加新标签----------#######
        tal_list = [
            #filename       tab name             classname
            # ("TSP_adapt",   "💬 后台环境适配",     "TspAdapt"),
            ("GPS_correct",  "📋 GPS校正",         "GpsCorrect"),
            # (""             "⚙️ 待添加",          self.create_history_tab),
        ]
        # 👇 导入外部标签页
        for filename, name, classname in tal_list:
            module = importlib.import_module(filename)#动态导入依赖模块# from TSP_adapt import TspAdapt # from GPS_correct import GpsCorrect
            creator = getattr(module, classname)
            widget = creator()
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.tabs.addTab(widget, name)


        # for name, creator in tal_list:
        #     widget = creator()
        #     widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #     self.tabs.addTab(widget, name)

        ######----------增加新标签----------#######


        # self.send_button.clicked.connect(self.send_message)  # 发送按钮绑定send_message
        # self.input_box.returnPressed.connect(self.send_message)  # 回车件绑定send_message


        # ========== 标签页 2：消息历史 ==========
    def create_history_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.history_display = QTextEdit()
        self.history_display.setReadOnly(True)
        layout.addWidget(self.history_display)
        widget.setLayout(layout)
        return widget




    #--------------通用接口--------------##
    def send_message(self):
        msg = self.input_box.text().strip()
        if msg:
            try:
                self.sock.send(msg.encode('utf-8'))
                self.append_message(f"我: {msg}")
                self.input_box.clear()
            except Exception as e:
                self.append_message(f"发送失败: {e}")

    def append_message(self, text):
        self.text_display.append(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClientApp()
    window.show()
    sys.exit(app.exec_())