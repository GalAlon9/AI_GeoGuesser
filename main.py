import requests
import random
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D



col_max_lat = 40.99460
col_min_lat = 36.99737
col_max_lon = -102.08537
col_min_lon = -109.05336

wyo_max_lat = 45.00122
wyo_min_lat = 40.99840
wyo_max_lon = -104.05769
wyo_min_lon = -111.05686

kan_max_lat = 40.00317
kan_min_lat = 36.99640
kan_max_lon = -95.31045
kan_min_lon = -102.05129

newMexico_max_lat = 36.99897
newMexico_min_lat = 32.00143
newMexico_max_lon = -103.00196
newMexico_min_lon = -109.04674

API_KEY = '' #insert your API key here

url = 'https://maps.googleapis.com/maps/api/streetview'

street_view_image_metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata" 


def get_random_location(min_lat, max_lat, min_lon, max_lon):
    rand_lat = random.uniform(min_lat, max_lat)
    rand_lon = random.uniform(min_lon, max_lon)
    return rand_lat, rand_lon
    

def get_random_picture(lat,lon,lable,location):
    #try to find a picture with 50 meters radius
    loc = str(lat) + ',' + str(lon)
    radius = 10000
    key = API_KEY

    params = {
       'location': loc,
        'radius': radius,
        'size': '600x300',
        'key': key,
        'return_error_code': 'true'
    }


    response = requests.get(url, params=params)
    

    if response.status_code == 404: 
        print("Invalid coordinates")
        return 0
    else:
        print("Valid coordinates")
        #save the picture with the label "lable.jpg" at the location database/location
        with open('database/'+location+'/'+str(lable)+'.jpg', 'wb') as f:
            f.write(response.content)

        #print the image coordinates
        print(loc)

        
            
        print("Done")
        return 1

def create_database():
    images = []
    labels = []
    X = np.zeros((1500,128,128,3), dtype=np.uint8)
    y = np.zeros((1500,1), dtype=np.uint8)

    kansas_folder = 'database/kansas/'
    wyoming_folder = 'database/wyoming/'
    colorado_folder = 'database/colorado/'
    
    for folder, label in zip([kansas_folder, wyoming_folder, colorado_folder], ['kansas', 'wyoming', 'colorado']):
        for filename in os.listdir(folder):
            if filename.endswith('.jpg'):
                img = Image.open(folder + filename)
                img = np.array(img)
                images.append(img)
                labels.append(label)

    X = np.array(images)

    y = np.array(labels)

    #convert the labels to one hot encoding
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y = to_categorical(y_encoded)

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # #check that it worked as expected
    # image_to_show = images[0]
    # with Image.fromarray(image_to_show) as img:
    #     img.show()
    # print(y[0])

    #save the data
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)


def CNN_model_train():
    #load the data
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    input_shape = (300, 600, 3)
    model = Sequential()

    #add convolutional layer with 32 filters, a 3x3 kernel, and relu activation function
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    #add a max pooling layer with a 2x2 pool size
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #add another convolutional layer with 64 filters and a 3x3 kernel
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    #add another max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #flatten the output of the convolutional layers
    model.add(Flatten())
    #add a dense layer with 128 neurons and a relu activation function
    model.add(Dense(128, activation='relu'))
    #add the output layer with 3 neurons and a softmax activation function
    model.add(Dense(3, activation='softmax'))
    #compile the model using categorical_crossentropy loss, and the adam optimizer and accuracy metrics
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    #train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4)
    #save the model
    model.save('CNN_model.h5')


def small_model_test():
    #load the model
    model = load_model('CNN_model.h5')

    print("Model loaded")

    #chose a random state from {kansas, wyoming, colorado}
    state = random.choice(['kansas', 'wyoming', 'colorado'])
    if state == 'kansas':
        lat, lon = get_random_location(kan_min_lat, kan_max_lat, kan_min_lon, kan_max_lon)
    elif state == 'wyoming':
        lat, lon = get_random_location(wyo_min_lat, wyo_max_lat, wyo_min_lon, wyo_max_lon)
    else:
        lat, lon = get_random_location(col_min_lat, col_max_lat, col_min_lon, col_max_lon)

    print("chose a random location in " + state + "")

    #get a random picture
    success = 0
    while success == 0:
        success = get_random_picture(lat, lon, state, "test")
    
    #load the picture
    img = Image.open('database/test/'+state+'.jpg')
    
    #convert the picture to an array of size (300,600,3)
    img = np.array(img)
    img = img.reshape(1,300,600,3)

    

    #predict the state
    prediction = model.predict(img)

    prediction = convert_prediction_to_state(prediction)

    print("The state is: " + state)
    print("The prediction is: " + str(prediction))


def test_model():
    #load the model
    model = load_model('CNN_model.h5')

    print("Model loaded")

    counter = 0
    #repete 100 times
    for i in range(100):
        print("test number: " + str(i))
        #chose a random state from {kansas, wyoming, colorado}
        state = random.choice(['kansas', 'wyoming', 'colorado'])
        min_lat, max_lat, min_lon, max_lon = 0, 0, 0, 0
        if state == 'kansas':
            min_lat, max_lat, min_lon, max_lon = kan_min_lat, kan_max_lat, kan_min_lon, kan_max_lon
        elif state == 'wyoming':
            min_lat, max_lat, min_lon, max_lon = wyo_min_lat, wyo_max_lat, wyo_min_lon, wyo_max_lon
        else:
            min_lat, max_lat, min_lon, max_lon = col_min_lat, col_max_lat, col_min_lon, col_max_lon
            
        lat, lon = get_random_location(min_lat, max_lat, min_lon, max_lon)
        #get a random picture
        success = 0
        trys = 0
        while success == 0:
            success = get_random_picture(lat, lon, state, "test")
            trys += 1
            if trys > 4:
                lat, lon = get_random_location(min_lat, max_lat, min_lon, max_lon)

        
        #load the picture
        img = Image.open('database/test/'+state+'.jpg')
        
        #convert the picture to an array of size (300,600,3)
        img = np.array(img)
        img = img.reshape(1,300,600,3)

        
        #predict the state
        prediction = model.predict(img)

        prediction = convert_prediction_to_state(prediction)

        

        if prediction == state:
            counter += 1
        
            

    print("The accuracy is: " + str(counter/100))


def convert_prediction_to_state(prediction):
    #find the index of the highest value in the prediction
    index = np.argmax(prediction)

    #convert the index to a state
    if index == 0:
        return 'colorado'
    elif index == 1:
        return 'kansas'
    else:
        return 'wyoming'
   

def main():
    # save 100 pictures labeled 0 to 99
    counter = 100
    while counter < 500:
        lat, lon = get_random_location(wyo_min_lat, wyo_max_lat, wyo_min_lon, wyo_max_lon)
        if get_random_picture(lat, lon, counter,"wyoming") == 1:
            counter += 1
            print(counter)

    create_database()

    CNN_model_train()
    small_model_test()
    test_model()
    model = load_model('CNN_model.h5')
    print(model.summary())

    
    



    






if __name__ == '__main__':
    main()



# def create_random_credentials():
    