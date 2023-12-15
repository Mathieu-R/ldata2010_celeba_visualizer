UNZIP := $(shell command -v unzip 2> /dev/null)

DATASETS_URL := 'https://uclouvain-my.sharepoint.com/personal/victor_joos_uclouvain_be/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fvictor%5Fjoos%5Fuclouvain%5Fbe%2FDocuments%2FLDATA2010%2Fceleba%2Ezip' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' -H 'Accept-Language: en-GB,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br' -H 'Referer: https://uclouvain-my.sharepoint.com/personal/victor_joos_uclouvain_be/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fvictor%5Fjoos%5Fuclouvain%5Fbe%2FDocuments%2FLDATA2010%2Fceleba%2Ezip&parent=%2Fpersonal%2Fvictor%5Fjoos%5Fuclouvain%5Fbe%2FDocuments%2FLDATA2010&ga=1' -H 'DNT: 1' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: iframe' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Connection: keep-alive' -H 'Cookie: MSFPC=GUID=cfd3872b330c49ebb54b9ea40dbd2ca3&HASH=cfd3&LV=202305&V=4&LU=1683631767027; MicrosoftApplicationsTelemetryDeviceId=94567cca-a5fe-4841-9ac1-415bb2386013; FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjEzLDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uI2UzNjAxNmUzNDRhNTAwNDJjZWJmYTJjN2IxY2NhNDg0ODAxMGU3MTAyMjJhMTUwMTcyMzhiODY3NzY0YWExZDUsMCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWFub24jZTM2MDE2ZTM0NGE1MDA0MmNlYmZhMmM3YjFjY2E0ODQ4MDEwZTcxMDIyMmExNTAxNzIzOGI4Njc3NjRhYTFkNSwxMzM0NzExMzEyNzAwMDAwMDAsMCwxMzM0NzE5OTIyNzE5NTMxNzYsMC4wLjAuMCwyNTgsNWE3MmQ3YjAtYTdiOS00YjU0LTkwZjgtNTgxYzk0Zjg1MTI0LCwsMWFmNmY3YTAtZDA4Yi03MDAwLWM1ZDEtYTA5YWE2NDgyODljLDFhZjZmN2EwLWQwOGItNzAwMC1jNWQxLWEwOWFhNjQ4Mjg5Yyw3VFp3ZXNFMWsweXM3MURxZXg0TXFRLDAsMCwwLCwsLDI2NTA0Njc3NDM5OTk5OTk5OTksMCwsLCwsLCwwLCwxODkwMzIsR0FkeFdYM3FnLXBsUDRlOVhCUDF5MTZpZmpVLERXbytLQWVyU0ZPbStmQnI0VTdPd1lkdXVCNVZMSHovV1hndXYzdklhNjBtRXpqQXM0RVBtajdDdWluL01mNytodkplYjZnQUZoYjhFd1U4U0NreER6WW5vVVFkajZlQXNGSllZcHcvbDU1aGJNODR3aS9xTHFIV3FMd0pNdDZrWnVHei9kazRBSG1oaGlCY2MwV29GV055OFA3Mi9tRUhiY25iNm5xVUlEZzhUME1iVktlZ2RoVE5neWxUelhDaDBmalhaVHVFcDdqQmZIeEdVKytkbkdscmM3QUZSUmU4akNEcjd0SjNHNi8xT3BaRlZRZG5qeEVLbHZIOGcrMHRkNGY0ekZHMkEvSExEdFhyZ0d1Yk9LUnIwWlNsTWVubnpHcEJ0SHlsSVZDaU9yeTd6Qk05NTRFY3FWWGlnZ3pFTllIRzNYSmFhK1FIVGJ1S1YxbWxCQT09PC9TUD4=; ai_session=EYiDl/bDMX8v/kndI03d1z|1702639236895|1702639282111'

DATA_FOLDER := data
ASSETS_FOLDER := assets 

resources:
	@if [ -z $(UNZIP) ]; then echo "Unzip could not be found. Please install it"; exit 2; fi
	curl $(DATASETS_URL) --output celeba.zip
	mkdir $(DATA_FOLDER)
	$(UNZIP) celeba.zip 
	mv celeba_buffalo_l.csv $(DATA_FOLDER)/celeba_buffalo_l.csv 
	mv celeba_buffalo_s.csv $(DATA_FOLDER)/celeba_buffalo_s.csv
	mv img_celeba $(ASSETS_FOLDER)/img_celeba
	rm -rf celeba.zip

precompute:
	python precompute_ndarray.py

run:
	python main.py