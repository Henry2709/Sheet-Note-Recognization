# Sheet-Note-Recognition

Digital Image Processsing Final Project

Sheet Music Intelligent Reader & Keyboard Simulator (SMIRKs)

Folder Structure:
```bash
├── Database    % Database(captured notes) used for Machine Learning/Deep Learning
│   ├── eighth_note
│   ├── half_note
│   ├── quarter_note
│   ├── test\ eighth_note
│   └── test\ quarter_note
├── GUI         % Graphical User Interface
│   ├── Gui.py
│   ├── SMIRKs_head.py
│   └── SMIRKs_head.pyc
├── Intelligent\ Reader\ using\ DIP      % Algorithm that we use in GUI
│   ├── SMIRKs.py
│   ├── SMIRKs_head.py
│   └── SMIRKs_v2.py
├── Intelligent\ Reader\ using\ ML\ (in\ progress)   % Algorithm that is still in progress
│   ├── Pic_segment.py
│   ├── SIFT.py
│   └── __init__.py
├── Keyboard\ Simulator     % Compile music and output an WAV file, then play the file
│   ├── Convert.py
│   └── test.wav
├── README.md
└── Sheet\ Music\ for\ Presentation   % Sheet Music used for presentation
    ├── Bolero.png
    ├── Symphony\ No.9.png
    └── Twinkle\ Twinkle\ Little\ Star.png
```

Instructions:

Basically, you can check the function of SMIRKs by launching /GUI/Gui.py.
Before running, please make sure you have installed all the required dependencies (which you can find in Gui.py).
Gui.py is compatiable with Python 2.7. If you want to run it with Python 3, please change Tkinter in GUi.py to tkinter.


