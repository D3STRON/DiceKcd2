# Dice From Kingdom Come Deliverance 2

## Installation

1) Clone this Repository
2) Make your virtual environment
`python -m venv venv`
    * Linux `source venv/bin/activate`
    * Windows `venv\Scripts\activate`
3) Install python dependencies 
`pip install -r requirements.txt`
4) `cd Streamlit`
5) Make sure by running below code seperately that the application can access your phones camera or camera source that can point to the dice for mac both phone and laptop should be on the same aple account
- `imoprt cv2`
- `cap = cv2.VideoCapture(1)` 

6) Run the streamlit using `streamlit run app.py`

## Gameplay
1) Start with selecting the game name this will create a new `<game_name>.pkl` file in the `./games` directory. To start a new game either delete the file or put in a new game name.

![alt text](./images/roll.png)


