import {Module} from "@nestjs/common";
import {MinioService} from "./minio.service";
import {NestMinioModule} from "nestjs-minio";
import {MinioController} from "./minio.controller";

@Module({
    imports: [
        NestMinioModule.register({
            endPoint: process.env.MINIO_HOST,
            port: 9000,
            useSSL: false,
            accessKey: process.env.MINIO_ROOT_USER,
            secretKey: process.env.MINIO_ROOT_PASSWORD
        })
    ],
    controllers: [MinioController],
    providers: [MinioService],
    exports: [MinioService]
})
export class MinioModule {
}