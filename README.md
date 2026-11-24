# Future Frame Prediction

DATA 785 project- Shaneel Patel

Tries to predict the next frame from 5 previous frames of moving squares.
Has an RNN and a Transformer. Diffusion model is also in the code but I mostly used the other two.

## Run
pip install -r requirements.txt
mkdir artifacts
cd src
python train.py --model rnn --epochs 4
python generate.py --model rnn

Data is made on the fly.. Models only predict one frame right now.
