# Mapping the Cradle to Prison Pipeline - Handwriting to Text

## Background
Translate 200 handwritten surveys from Phase 2 into CSV file and do an analysis on Phase 1 (given) and Phase 2 data.

## Approach
### We seperated the survey into two tasks: Checkbox extraction and Plain text extraction
1. Use AWS texttract (https://aws.amazon.com/textract/) to extract the plain text into a CSV file 
2. Use OpenCV (https://docs.opencv.org/4.x/index.html) to extract the checkbox data into a CSV file
3. Combine the results of both CSV files into one final CSV
4. Analyze the data using python
![img6](/images/img6.png)

## Instructions
* Plain text model:
    1. Set up environment
        - `pip install boto3` 
        - `aws configure`
        - `aws_access_key_id = YOUR_ACCESS_KEY`
        - `aws_secret_access_key = YOUR_SECRET_KEY`
        - `region=us-east-1`
    2. Create a s3 bucket and upload files 
    3. Run 'Texttract.ipynb' on the bucket
    4. If using spanish survyes, run 'Texttract_Spanish.ipynb'

* Checkbox model: (To avoid dependency issues run in Google Colab)
    1. In 'checkbox_model.ipynb', modify cell 6 to be the file path where the PDFs are located
        ![img4](/images/img4.png)
        - Should end with *.pdf like picture shows
    2. Modify the last cell to have the file path of where you want to save the output CSV file
        ![img5](/images/img5.png)
    3. Run all cells and wait for the final output

## Bugs and Errors
1. Both the plain text and checkbox models do not achieve 100% accuracy. Therefore, human review is needed to verify the results of both models.
2. Nested questions (questions inside another question) do not work at all with the checkbox model (see example below)
![img7](/images/img7.png)

## Deployment
In order to add new surveys:
1. Upload to the s3 bucket
2. Run 'Texttract_continued.ipynb' on the new file

## Analysis
Results from some of our analysis
![img1](/images/img1.png)
![img2](/images/img2.png)
![img3](/images/img3.png)

