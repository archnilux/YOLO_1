if you are new to these tasks follow the steps below one by one:

1- download all code files with the .py extension and the requirements.tx file.
2- create an empty folder named YOLO (you can change the name) on your computer. inside the YOLO folder create a second folder named src.
3- move the inference.py, train.py and utils.py files into the src folder. main.py, demo.py and requirements.txt should remain in the YOLO folder. the structure should look like this:

YOLO/
├── main.py
├── demo.py
├── requirements.txt
├── src/
│   ├── train.py
│   ├── inference.py
│   └── utils.py

4- click the bar at the top of the YOLO folder where it says C:\Users\UserName\Desktop\YOLO. type `cmd` there and press enter. a black screen will appear. you will copy and paste the commands i will provide below into that black screen and press enter.

5- paste the command `pip install -r requirements.txt` and press enter. this command will install the dependencies i put in the requirements.txt file on your computer. wait until the installation is finished.

6- paste the command `python main.py --mode prepare` and press enter. this command will download and prepare the dataset. after the process is complete if you check the YOLO folder you will see that new folders have been added. the structure will be as follows:

YOLO/
├── main.py
├── demo.py
├── requirements.txt
├── src/
│   ├── train.py
│   ├── inference.py
│   └── utils.py
├── models/
├── data/
└── results/

7- return to the black screen (console) and paste the command `python main.py --mode train --epochs 25` and press enter. this command will start the model training. depending on your computer hardware this may take a few minutes or a few hours.

ps: if you get error messages in the console starting with RunTimeError or ImportError it means there is a conflict among the installed packages on your computer. to solve the problem you can run the commands below but if you have other projects on your computer these commands may break their package compatibility. for this reason i recommend running the commands in an isolated conda environment or doing some research before executing them.

pip uninstall -y numpy torch torchvision torchaudio opencv-python-headless ultralytics
pip install numpy==1.26.4
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install opencv-python-headless==4.9.0.80
pip install ultralytics==8.2.50

after running these commands try running the command in step 7 again. most likely it will work without any problems.
