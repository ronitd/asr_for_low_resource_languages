import torch 
import csv


def load_obj(path):
    with open(name, 'rb') as f:
        return pickle.load(f)
def check_loss(loss, loss_value): 
	loss_valid = True 
	error = '' 
	if loss_value == float("inf") or loss_value == float("-inf"): 
		loss_valid = False 
		error = "WARNING: received an inf loss" 
	elif torch.isnan(loss).sum() > 0: 
		loss_valid = False 
		error = "WARNING: received a nan loss, setting loss value to 0"
	elif loss_value < 0: 
		loss_valid = False 
		error = "WARNING: received negative loss"
	return loss_valid, error 

def read_csv_deepspeech(csv_file, sorted_by_size=False): 
	raw_audio_paths = [] 
	transcripts = []
	durations = [] 
	languages = []	
	with open(csv_file) as f:
		temp_data = [] 
		csv_reader = csv.reader(f, delimiter=',') 
		for row in csv_reader:
			wav_path = row[0]
			if (wav_path != "wav_filename"):
				temp_data.append(row)

		#Sort by wavsize, which is row[1]
		if not sorted_by_size: 
			temp_data = sorted(temp_data, key=lambda k: int(k[1]))
		iter = 0
		for temp_datum in temp_data: 
			raw_audio_paths.append(temp_datum[0])
			durations.append(temp_datum[1])
			transcripts.append(temp_datum[2])
			#languages.append(int(temp_datum[3]))
			languages.append(1)	
			iter += 1
		languages = torch.LongTensor(languages)
		#print(languages[0])
		#print(languages.shape)
		one_hot_encoded_languages = torch.zeros(len(languages), languages.max()+1).scatter_(1, languages.unsqueeze(1), 1.)
		#print("One hot Language: ", one_hot_encoded_languages.shape
	return raw_audio_paths, transcripts, one_hot_encoded_languages

def read_csv_mfccs(csv_file):
	raw_mfccs_path = [] 
	transcripts = [] 
	with open(csv_file) as f: 
		csv_reader = csv.reader(f, delimiter=",") 
		for row in csv_reader: 
			wav_path = row[0] 
			if (wav_path != "wav_filename"): 
				raw_mfccs_path.append(row[2])
				transcripts.append(row[1])
	return raw_mfccs_path, transcripts 

def read_label_file(label_file): 
	labels = set()
	with open(label_file, 'r') as f: 
		for row in f: 
			labels.add(row[0])
	labels = sorted(labels)
	labels_str = ""
	for label in labels: 
		labels_str += label
	return labels_str


def levenshtein_distance(a, b): 
	#Compute levenshtein distance between 
	#array a and array b. 
	#If CER: a and b should be strings
	#If WER: a and b should be strings.split() 

	n, m = len(a), len(b) 
	if n > m: 
		a, b = b, a 
		n, m = m, n 

	current = list(range(n+1)) 
	for i in range(1, m+1): 
		previous, current = current, [i] + [0] * n 
		for j in range(1, n+1): 
			add, delete = previous[j] +1, current[j-1]+1 
			change = previous[j-1] 
			if a[j-1] != b[i-1]: 
				change = change + 1
			current[j] = min(add, delete, change) 

	return current[n]
