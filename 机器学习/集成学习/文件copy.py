with open("./images/0.9709465431428076.jpg","r") as f:
    content = f.read()
    with open("./result2/result.JPG","w") as f:
        f.write(content)
