import glob

textstr = '#text='
fils = glob.glob("./tabnak-dataset/*")
with open('dataset/data.txt', 'w') as dataset:
    for file in fils:
        with open(file) as fr:
            for line in fr.readlines():
                if textstr in line:
                    dataset.writelines(line.replace(textstr, ''))
            fr.close()
    dataset.close()
