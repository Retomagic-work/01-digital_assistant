# Requests

## Create a new request

**POST** - `/requests`

### Headers

```json
{
  "Content-Type": "multipart/form-data"
}
```

### Body

```ts 
interface CreateRequestBodyDto {
    files?: Array<File>, // Array of files (max files 10)
    [key: string]: String // Other dynamic fields
}
```

### Response

```json 
{
  "id": 1,
  "status": "pending",
  "original_images": [
    "http://localhost/files/requests/30d292fc80164b676b60f1c39b04731a.jpeg"
  ],
  "finished_images": [],
  "data": {
    "test": "20"
  },
  "created_at": "2023-10-10T01:00:41.851Z",
  "updated_at": "2023-10-10T01:00:41.879Z"
}
```

## Get request by ID

**GET** - `/requests/:id`

### Response

```json 
{
  "id": 1,
  "status": "success",
  "original_images": [
    "http://localhost/files/requests/30d292fc80164b676b60f1c39b04731a.jpeg"
  ],
  "finished_images": [],
  "data": {
    "test": "20"
  },
  "created_at": "2023-10-10T01:00:41.851Z",
  "updated_at": "2023-10-10T01:00:41.879Z"
}
```