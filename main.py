from options import get_args
from train import FlareModel


def main():
    opt = get_args()    

    try:

        train_model = FlareModel(opt)

        if opt.load_model:
            train_model.load_model()
        
        print("Model training started")
        train_model.train()
        train_model.save_model()

    except KeyboardInterrupt:
        print("Keyboard Interrupt: Saving existing model !!")
        train_model.save_model()


if __name__ == '__main__':
    main()

    
