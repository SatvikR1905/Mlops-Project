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
