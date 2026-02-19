# Mlops-Project

## Project Structure

```
│
├── airflow/                     
│   ├── dags/                    
│   ├── logs/                    
│   ├── plugins/                 
│   ├── docker-compose.yml       
│   └── Dockerfile               
│
├── data/
│   └── processed/               
│       ├── train.csv
│       └── test.csv
│
├── src/
│   └── c/                      
│       ├── preprocess.py        
│       ├── train.py             
│       ├── model.py             
│       ├── convert_to_onnx.py   
│       └── inference.py         
│
├── models/
│   └── v1/                      
│       ├── model.pt             
│       ├── model.onnx           
│       └── model.onnx_data      
│
├── k8s/                         
│   └── deployment.yaml          
│
├── monitoring/                  
│   └── prometheus.yml           
│
├── Dockerfile                   
├── requirements.txt             
├── README.md                    
└── .gitignore                   
```

# Airflow Result
![WhatsApp Image 2026-01-30 at 1 38 15 AM](https://github.com/user-attachments/assets/01f99450-abc7-49d5-bee8-c363ecb5534a)

