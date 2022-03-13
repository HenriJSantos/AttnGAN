import cv2
import h5py
import pickle

DATA_PATH = "../data/fashiongen/"

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def clean_string(orig):
    new = str(orig)[2:len(orig)-1]
    return new

def create_dataset(file, isTrain):
    total_len = len(file["index"])
    filenames = []
    print("Creating " + ("train" if isTrain else "test") + " dataset.")
    for n in range(len(file["index"])):
        filename = "FashionGEN_" + ("train_" if isTrain else "test_") + str(n).zfill(8)
        filenames.append(filename)
        cv2.imwrite(DATA_PATH + "images/"+filename+".png", cv2.cvtColor(file["input_image"][n], cv2.COLOR_RGB2BGR))
        f = open(DATA_PATH + "text/" + filename + ".txt","w+")
        f.write(clean_string(file["input_description"][n][0]))
        f.close()
        printProgressBar(n, total_len, prefix='Progress:', suffix="Complete ("+str(n)+"/"+str(total_len)+")", length=50)

    with open(DATA_PATH + ("train/" if isTrain else "test/") + "filenames.pickle", "wb") as f:
        pickle.dump(filenames, f)

create_dataset(h5py.File(DATA_PATH + "fashiongen_256_256_train.h5", "r"), True)
create_dataset(h5py.File(DATA_PATH + "fashiongen_256_256_validation.h5", "r"), False)