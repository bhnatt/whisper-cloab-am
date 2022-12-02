#from google.colab import drive

from docx import Document
from docx.shared import Pt
import glob
import numpy as np
import os
import re
import time
import torch
import whisper
from whisper_utils import write_txt, write_srt

try:
    from google.colab import drive
    is_google_colab = True
except ImportError:
    is_google_colab = False


def mountDrive () :
    if is_google_colab == False :
        print ('Not a Google Colab environment !!!')
        return False

    google_drive = '/content/drive/MyDrive/'
    isExist = os.path.exists (google_drive)
    if not isExist :
        drive.mount('/content/drive/')
        print ('Google Driver mounted.')
    
    return True
###

            
class WhisperAM :
    def __init__ (self, model_name, data_dir) :
        self.model_name = model_name
        self.data_dir   = data_dir
        self.out_dir = self.data_dir
        
        isExist = os.path.exists (self.out_dir)
        if not isExist :
            os.makedirs (self.out_dir, exist_ok=False)
    ###


    ### https://betterprogramming.pub/openais-whisper-tutorial-42140dd696ee
    def checkCuda (self) :
        # make sure if it shows 'cuda' which means gpu ready
        # if not, change the runtime type : menu > runtime > change runtime type > select GPU, then start from the beginning again

        torch.cuda.is_available ()
        self.DEVICE = "cuda" if torch.cuda.is_available () else "cpu"
        print (self.DEVICE)


    #@title Whisper model selection : medium.en (default)
    def loadModel (self) :
        #model_name = 'tiny.en' #@param ["medium.en", "large", "tiny.en"]
        self.model = whisper.load_model (self.model_name, device=self.DEVICE)

        #model = whisper.load_model ('tiny.en', device=DEVICE)
        #model = whisper.load_model ('small.en', device=DEVICE)
        #model = whisper.load_model ('medium.en', device=DEVICE)
        #model = whisper.load_model ('large', device=DEVICE)
    ###


    def getText (self, segments) :
        text = []
        for segment in segments:
            text.append (segment['text'].strip())

        return '\n'.join (text)
    ###


    options = dict (language='English', beam_size=5, best_of=5)
    transcribe_options = dict (task="transcribe", **options)

    #@title transcribe function definition
    def transcribe (self, name, output_dir) :
        mp3_file = output_dir + name + '.mp3'

        self.result = self.model.transcribe (mp3_file, **self.transcribe_options)
        return self.result
    ###
        
    
    def saveResult (self, name, output_dir, result) :
        text = self.result ['text']
        segments = self.result ['segments']

        # save temp SRT : caption
        srt_file_tmp = output_dir + name + '.srt.tmp'
        with open (srt_file_tmp, "w", encoding="utf-8") as wf:
            write_srt (segments, file=wf)

        # postprocess srt
        with open (srt_file_tmp) as rf:
            text = rf.read ()
            text2 = self.postprocess (text)

        # save SRT : caption
        srt_file = output_dir + name + '.srt'
        with open (srt_file, "w", encoding="utf-8") as wf:
            wf.write (text2)
            os.remove (srt_file_tmp)
            
        # post processing : apply macros
        text = self.getText (segments)
        text2 = self.postprocess (text)

        txt_file = output_dir + name + '.txt'
        with open (txt_file, 'w') as wf :
            wf.write (text2)
            #os.remove (tmp_file)


        doc_file = output_dir + name + '.docx'
        self.saveToDocx (text2, doc_file)

        return result    
    ###



    #@title postprocessing : macro, save to docx

    # macro definition
    rdict = { '1000s': 'thousands', 'a collective consciousness': 'the collective consciousness', 'Ascended Master': 'ascended master', 'a golden age': 'the golden age', 'And so ': '', 'And so, ': '', 'and so forth and so on': 'and so forth', "aren't": 'are not', 'behaviour': 'behavior', "can't": 'cannot', 'Christ to it': 'Christhood', 'colour': 'color', 'conscious you': 'Conscious You', "couldn't": 'could not', 'defence': 'defense', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', 'Earth': 'earth', 'fall and beaks': 'fallen beings', 'flavour': 'flavor', 'freewill': 'free will', 'fulfil': 'fulfill', 'Golden Age': 'golden age', "haven't": 'have not', "he's": 'he is', "I'm": 'I am', "isn't": 'is not', "it's": 'it is', "It's": 'It is', 'karmic board': 'Karmic Board', 'labour': 'labor', 'offence': 'offense', 'out picture': 'out-picture', 'realise': 'realize', 'saviour': 'savior', 'self awareness': 'self-awareness', "shouldn't": 'should not', 'summit lighthouse': 'Summit Lighthouse', "that's": 'that is', "That's": 'That is', "there's": 'there is', "There's": 'There is', "they're": 'they are', "wasn't": 'was not', "we're": 'we are', "We're": 'We are', "weren't": 'were not', "we've": 'we have', "What's": 'What is', "what's": 'what is', "wouldn't": 'would not', "won't": 'will not', "you'll": 'you will', "you're": 'you are', "You're": 'You are', 'So ': '', 'So, ': '', 'Well ': 'Well,', 'separate cell': 'separate self', 'separate cells': 'separate selves' }


    @staticmethod
    def multiple_replace (adict, text):
        # Create a regular expression from all of the dictionary keys
        regex = re.compile ( "|".join (map (re.escape, adict.keys ( ))) )

        # For each match, look up the corresponding value in the dictionary
        return regex.sub (lambda match: adict [match.group (0)], text)
    ###


    def postprocess (self, text) :
        # macro processing
        callback = lambda pat: pat.group(2).upper()
        text2 = re.sub ('(So|So,|And so|And so,) ([a-zA-Z])', callback, text)

        text2 = self.multiple_replace (self.rdict, text2)

        return text2
    ###


    # save to docx file
    @staticmethod
    def saveToDocx (text, filename) :
        document = Document()
        para = document.add_paragraph ()

        # title
        title = '.'.join (filename.split ('/') [-1].split ('.') [:-1])
        run = para.add_run (title)
        run.font.size = Pt (16) # font size
        run.font.name = 'Arial' # font name
        run.bold = True

        # body
        body = document.add_paragraph ()
        run = body.add_run ('\n')

        sentences = text.split ('\n')
        run = body.add_run (' '.join (sentences))
        run.font.size = Pt (14) # font size
        run.font.name = 'Arial' # font name

        # save document
        document.save (filename)
    ###        


    #get input audio data list in 'data' folder
    def getFiles (self) :
        files1 = glob.glob (self.data_dir + '/*.mp3')
        files2 = glob.glob (self.data_dir + '/*.m4a')

        self.input_data = files1 + files2
        return self.input_data
    ###
    

    # main program
    def run (self) :
        self.loadModel ()
        input_data = self.getFiles ()
        
        for source_file_name in input_data :
            print (source_file_name)
            target_name = '.'.join (source_file_name.split ('/') [-1].split ('.') [:-1])

            # trascribing
            result = self.transcribe (target_name, self.data_dir)
            result = self.saveResult (target_name, self.data_dir, result)
            text = result ['text']
            print (target_name, '\n', text [:50], '...', text [-50:])
        ### for
    ### main
### class


def test () :
    #@title run Whisper

    data_dir = 'data/'

    ds = mountDrive ()
    google_drive = '/content/drive/MyDrive/' if ds else './'

    data_path   = google_drive + data_dir
    print (data_path)

    model_name   = 'tiny.en'
    
    am = WhisperAM (model_name, data_path)
    am.checkCuda ()

    print ('Start :', time.strftime('%X %x %Z'))
    am.run ()
    print ('End  :', time.strftime('%X %x %Z'))


    #@title unassign google colab resource
    #print ('Google Colab resource unassigned.')
    
    # shutdown colab runtime. works well
    #from google.colab import runtime
    #runtime.unassign()
###


if __name__ == '__main__' :
    test ()