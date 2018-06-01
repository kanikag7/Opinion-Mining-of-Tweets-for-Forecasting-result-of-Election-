from preprocess import preprocess
import xlrd 

def main():
    
    p=preprocess()
    #the name of the excel file
    book = xlrd.open_workbook("training-Obama-Romney-tweets.xlsx")
    words_filename=["obama_words.txt","romney_words.txt"]
    labels_filename=["obama_labels.txt","romney_labels.txt"]
    for i in range(0,2):
        worksheet = book.sheet_by_index(i)
        end_val=worksheet.nrows
        start_val=2
        
        
        
        #extract data and store it in file featurelist.txt
        p.extractdata(worksheet,start_val,end_val,words_filename[i],labels_filename[i])
        
main()

