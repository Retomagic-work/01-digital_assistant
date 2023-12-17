import {Injectable, HttpException, HttpStatus, Inject, NotFoundException} from "@nestjs/common";
import {BufferedFile} from "./dto/file.dto";
import * as crypto from "crypto";
import {MINIO_CONNECTION} from "nestjs-minio";
import {Client} from 'minio'

@Injectable()
export class MinioService {
    public bucketRequestsName = 'requests'
    public bucketResponseName = 'responses'
    public bucketDocumentsName = 'documents'

    constructor(@Inject(MINIO_CONNECTION) private readonly minioClient: Client) {
    }

    /* Create buckets */
    private async onModuleInit() {
        if (!(await this.minioClient.bucketExists(this.bucketRequestsName))) {
            this.minioClient.makeBucket(this.bucketRequestsName, () => {
                console.log('Bucket created: ' + this.bucketRequestsName)
            })
        }

        if (!(await this.minioClient.bucketExists(this.bucketResponseName))) {
            this.minioClient.makeBucket(this.bucketResponseName, () => {
                console.log('Bucket created: ' + this.bucketResponseName)
            })
        }

        if (!(await this.minioClient.bucketExists(this.bucketDocumentsName))) {
            this.minioClient.makeBucket(this.bucketDocumentsName, () => {
                console.log('Bucket created: ' + this.bucketDocumentsName)
            })
        }
    }

    public getUrlFile(filename: string, bucketName: string = this.bucketRequestsName): string {
        return `${process.env.SERVER_ADDRESS}files/${bucketName}/${filename}`
    }

    public async getFile(filename: string, bucketName: string = this.bucketRequestsName) {
        const file = await this.minioClient.getObject(bucketName, filename);
        if (!file) throw new NotFoundException();

        return file
    }

    public async upload(file: BufferedFile, bucket = this.bucketRequestsName) {
        let temp_filename = Date.now().toString();
        let hashedFileName = crypto.createHash("md5").update(temp_filename).digest("hex");
        let ext = file.originalname.substring(file.originalname.lastIndexOf("."), file.originalname.length);

        const filename = hashedFileName + ext;
        const fileBuffer = file.buffer;

        this.minioClient.putObject(bucket, filename, fileBuffer, function (err, res) {
            if (err) throw new HttpException("Error uploading file: " + err, HttpStatus.BAD_REQUEST);
        });

        return filename
    }
}
