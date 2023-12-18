to run:
pip install -r requirements.txt

put russian-troll .csv files in data/csv/ 

run using: 
python twatter.py --max=#tweets --train=bool --latent_dim=#latent_dimenstion --all=bool

Arg.parse descriptions:
parser.add_argument('--train', type=boolean_string, default='False', help='train Word2Vec embedder')
parser.add_argument('--test', type=boolean_string, default='False', help='test Word2Vec embedder')
parser.add_argument('--vanilla', type=boolean_string, default='False', help='test Word2Vec embedder')
parser.add_argument('--all', type=boolean_string, default='False', help='go for broke on the entire dataset')
parser.add_argument('--max', type=int, default=100, help='max size of training data')
parser.add_argument('--latent_dim', type=int, default=100, help='max size of training data')