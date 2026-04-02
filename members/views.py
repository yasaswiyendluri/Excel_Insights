from django.shortcuts import render
import pandas as pd

def upload_file(request):
    data_preview = None
    summary=None

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        try:
            df = pd.read_excel(file)
            data_preview= df.head().to_html(classes="table table-striped")
            summary={
                'Rows':df.shape[0],
                'Columns':df.shape[1],
                'Columns_Names':','.join(df.columns),
                'Missing_values':df.isnull().sum().to_dict(),
            }
        except Exception as e:
            print("Error reading file:", e) 

    return render(request, 'upload.html', {'data': data_preview,'summary':summary})