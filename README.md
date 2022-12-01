# MediBOX
## An  Integrated Web Application for Personalised Medical Assistance
> This is a Flask Application. To run this application you must have python installed on your system.
> To install and run flask application have a look at [flask's](https://github.com/pallets/flask) official documentation
### Introduction
###### Health is a crucial facet of everybody's life. We all have to take medicine prescribed by doctors from time to time to sustain the soundness of our bodies. It is coherently laborious to maintain track of all the medications throughout life. Dissimilarity exists among individuals and medical institutions for treating patients with diverse medicines. Specific drugs might treat a patient positively while they might not turn out to be that effective on others. Hence, understanding each patient's body and the side effects of the drugs on them is crucial for doctors. The work of the doctors can be eased if they can refer to the patient's medical records. On the contrary, it is a tedious and meticulous job for the patients to maintain their own drug history. The proposed model Medibox will become a bridge for the doctors and the patient's smooth communication. The users can easily upload their prescriptions that will be converted into a digital repository along with more features. Overall, Medibox is a one-spot solution for patients to start pursuing a wholesome life and to keep a record of their medical history effortlessly.
### Current Senario 
###### Nowadays, we have seen that most people end up taking several medications throughout their lifespans. Sometimes, there might be a possibility that the disease might relapse back, instead of visiting the doctor again one can simply refer to the previous prescriptions. But in such situations, people might not find that particular prescription from the pile, or the chances are that the print of that prescription faded away which might cause difficulty in reading the prescribed medication. With the developing technology in the health sector, maintaining health records is significant in the upcoming days. Although digital logs are available, only the doctors are permitted to access the repository making it unreachable for the patients. However, there are many portals available in today's world, but they are missing some functionalities. Other than this, the portal can directly connect with the hospitals and clinics so that users might be able to access their data. But this will create a major concern as the hospital's all private data will get exposed to the outer world which can lead to vulnerable circumstances. Having a third-party portal such as Medibox is helpful as the user has the right to show only the important data without worrying about any privacy breach.

### What is MediBOX ?
###### Medibox is a multi-modular online medical platform where a patient simply needs to upload a scanned prescription and the web application will perform the remaining tasks by extracting the essential details from the prescription, generating a handy electronic medical records repository that can be easily accessed at any point in time.

###### It also provides convenience by offering features like finding nearby hospitals or doctors. Moreover, the application focuses on the well-being of the user by estimating their body mass index concerning their present-day height and weight. According to their BMI, Medibox will also suggest relative videos to the user for a better understanding of their situation and to develop a path to healthy living. For smooth navigation through the prescriptions and medications, Medibox also assists in presenting a chatbot for the users. A chatbot is a program that is designed to imitate intelligent dialogue over text or speech. These systems can self-learn and restore their knowledge with the help of humans or web resources. Because knowledge is saved in advance, this application is extremely important. To respond to user inquiries, the system application engages the question-and-answer protocol in the form of a chatbot. Apart from this, user can even schedule their medication timings flexibly at any date and time.  Medibox also provides a calendar for patients to remind them about their medications periodically. Hence, Medibox is a collection of modules that will induce the users to a better, fitter, and easily calculated healthy lifestyle.

### Prerequisites
###### Before moving further let us discuss some of the concepts which are required to get a better understanding of the architecture.
 **Image Preprocessing:**  
###### The goal of pre-processing is to improve image quality so that it can be analysed more effectively. By suppressing unwanted distortions and enhancing some features that are required for the application one can get a better and faster outcome, as less resources will be used. These characteristics may differ depending on the application. We are preprocessing the prescriptions to extract accurate data from it. As all the data is fetched depends on the image, which makes it not only the initial but the key stage of the process. 
**Pytesseract:**
###### Python-tesseract is an OCR engine wrapper for Google's Tesseract. It can also be used as a standalone tesseract invocation script because it can read all image types such as jpeg, png, gif, bmp, tiff, and others. Furthermore, when used as a script, Python-tesseract will print the extracted text as string rather than storing it to a file.
**Named Entity Relationship (NER):**
###### NER entails identifying key information in a text and categorizing it into a set of predetermined categories. An entity is essentially the object that is mentioned or referred to repeatedly throughout the text. NER is a two-step process: Identifying entities in the text and sorting them into various annotations such as a person, names, locations, organizations, and so on. However, we have customized the model which can predict the drug names accordingly. Figure  shows the in-depth process of the custom NER model. At first, tokens from the text are vectorized to speed up the process, these vectors are then labeled with the help of an annotator, and the vectors are analyzed by the parser and further categorized by the lemmatizer.
![ NER Pipeline ](https://d33wubrfki0l68.cloudfront.net/17030069c6d34a9b6fa43370ab683dd9f2f286ab/5b941/pipeline-design-b5ec1ba0f7a242d901ada88fa36b1002.svg) <br>
**Web Scrapping:** 
###### Web scraping is a technique for extracting information from web pages. The data can be extracted manually or with the help of tools. Web scraping is a relatively recent technique that has been utilised for a variety of reasons throughout history. In this project we will be using web scraper in the chatbot module. We used beautifulsoup to extract data from websites to provide accurate response to user queries.
 <sub> *note: Scrapped data from the public website as we had no access to medical data. Still working on getting access to websites data with more ethical means. </sub>
 
 ### Drug Dataset
 ###### Used the dataset available from [drugs.com](https://www.drugs.com/), where drug names were used as annotations and then they were passed to reviews as our training data. Based on existing work and our research we found this dataset to be more suitable for the prediction of drugs from prescription. [Drugs.com](https://www.drugs.com/) is a website where we can find all kinds of drug names which help segment it from the prescription. It consists of about 3700 drugs names. Other than drug names, it also has reviews that are used to train the [Spacy](https://spacy.io/) model.
 
 ### Proposed Approach
 ###### Below figure shows the methodology followed to run each fuctional components of the application.
 ![Proposed Model](/static/images/block.png)
 
 ##### DashBoard
###### Medibox consists of multiple modules that assist the user in maintaining a good healthy life. The user will upload their prescription after logging into the dashboard. Dashboard will display the key statistics gathered from patients previously uploaded prescriptions and will also visualize fluctuations in their BMI value. Apart from the graphical data it will also provide direct access to patients latest prescriptions. Following image is a snapshot of dashboard of the application where users can not only find and download their latest prescriptions but can get a direct insight on their visually appealing health statistics.
![Dashboard](/static/img/dashboard.png)
 
 * **Image Preprocessing:** 
###### The process starts with capturing an image of the printed prescription as an input which includes medical information. For better prediction of an image as a string, An image is passed for the image preprocessing. It is the process of resizing the images, and removing the noises for better prediction. In image preprocessing, initially, the image is converted from RGB to gray-scale. The gray-scale image is used for image binarization. The process of converting a multi-tone image to a bi-tonal image is known as binarization. It is common to practice in the case of document images to map foreground text pixels to black and the rest of the image (background) to white. The resultant image is used for image dilation. Dilation is used to expand an element A by using a structuring element B. This process adds pixels to the boundaries of objects. The final image is then passed for contour detection wherein the whole image is segregated into different parts or regions. In this different region, contours are recognized as continuous lines that covers the boundary of that object. Following image is the sample of preproccessed prescription format that the application has used for training and testing purpose.
![Preprocessed prescription](/static/img/prescriptionFormat.jpg)

* **Optical Character Recognition (OCR):**
###### The resultant image from preprocessing is taken as an input of optical character recognition (OCR) using Pytesseract. Pytesseract, also known as Python-tesseract, is a Python-based OCR tool that serves as a wrapper for the Tesseract-OCR Engine. After performing OCR, the output string is passed to the Named Entity-Relationship (NER) Model. As discussed in the Prerequisites, the NER model is used to predict the drug's name from the string that is the outcome of the OCR. To do so, the NER model is trained on the reviews of the dataset which will predict the drugs. Initially, the dataset is cleaned by removing the unwanted data and focuses mainly on the reviews. For the training of the model, drug names are used as annotations and are mapped with respect to reviews for prediction. 

###### In addition to drugs, the module will also detect data such as the doctor's name, doctor's number, hospital address, prescription date, and the dosage of medication from the uploaded prescription. Medibox also stores the original image of the prescription for the user to reduce the burden of keeping all the physical copies of the prescriptions at one place. The user can easily access the digital copy from the website which is more handy and convenient. After fetching the address, the application can direct the users to their doctor's location . This is achieved using [Google Maps API](https://developers.google.com/maps). Additionally, with the help of prescription date and drug dosage a medication scheduler is implemented. [Google Calendar API](https://developers.google.com/calendar/api) is used to schedule the date and time of the prescription and drugs accordingly. The website will set a reminder message for the user to take prescribed medicines on time. Since, we are using Calender's API, the schedule will also reflect on their google calendar without the need to open the application.

* **BODY MASS INDEX (BMI):**
###### Moreover, the application also has a feature to calculate the user's Body Mass Index (BMI) which is a valuable tool for determining whether or not someone is overweight or obese. It is calculated from the weight and height of the user. It is recommended by WHO to have the body composition assessed every month to keep ourselves fit and healthy. Indeed, knowing the BMI can help the patient and their doctor to determine any health risks that may be possible. Users can regularly update their height and weight to measure their BMI. A visually appealing graph is generated to represent the varying BMIs periodically. Furthermore, corresponding to the BMI results, the module also suggests related videos to maintain a healthy lifestyle using [Youtube API](https://developers.google.com/youtube/v3).

* **Medical ChatBot:**
###### Another functionality of this web application is the Medical Chatbot. The AI-based personal medical assistant is used to assist the user throughout the application. The chatbot can attempt to answer your queries regarding general health issues based on the keywords provided. The chatbot is developed on an intent file that is utilized to answer the user in a certain way. The intent file consists of a bunch of questions and medical word patterns that the user might ask. These questions are mapped to a set of relevant answers in JSON format. The group to which each input belongs to is indicated by a tag corresponding to their category in the file. The intent file is trained on a neural network to classify a phrase of words as one of the tags in the dataset. More the number of tags, responses, and patterns can be provided to the chatbot, the better and more complex it would be.

###### Sometimes there are chances that users may ask some questions for which the chatbot is not trained. For such a situation, we are using a method called Web Scraping. Whenever a user asks such queries which are not answerable by the chatbot, it will fetch the answer from the government medical website [National Health Service](https://www.nhs.uk/medicines/) to reply to the user's query all of which is implemented using web scraping. We are using web scraping only for test purposes as we don't have border access to healthcare data at this moment.

* **Nearby Hospitals and Clinics:**
###### In order to locate the hospitals or clinics in the surrounding region of the user, we have incorporated another component namely Nearby Clinics using [Google Places API](https://developers.google.com/maps/documentation/places/web-service/overview). All the details of the doctor and their user ratings along with the directions to their hospitals from the user's current location. To use this component, the user is first required to provide access to its current location. The current location is determined based on its latitude and longitude which is then passed onto the API to fetch the nearby health units within the radius of 5km.

