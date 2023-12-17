import {
    Controller, Get, Param, Res,
} from "@nestjs/common";
import {Throttle} from "@nestjs/throttler";
import {MinioService} from "./minio.service";

@Controller("files")
export class MinioController {
    constructor(private readonly minioClientService: MinioService) {
    }

    @Get('requests/:filename')
    @Throttle(60, 60)
    async getRequestsFile(@Param('filename') filename: string, @Res() res: any) {
        try {
            const response = await this.minioClientService.getFile(filename, this.minioClientService.bucketRequestsName);
            response.pipe(res)
        } catch (e) {
            throw e;
        }
    }

    @Get('responses/:filename')
    @Throttle(60, 60)
    async getResponsesFile(@Param('filename') filename: string, @Res() res: any) {
        try {
            const response = await this.minioClientService.getFile(filename, this.minioClientService.bucketResponseName);
            response.pipe(res)
        } catch (e) {
            throw e;
        }
    }

    @Get('documents/:filename')
    @Throttle(60, 60)
    async getDocumentsFile(@Param('filename') filename: string, @Res() res: any) {
        try {
            const response = await this.minioClientService.getFile(filename, this.minioClientService.bucketDocumentsName);
            response.pipe(res)
        } catch (e) {
            throw e;
        }
    }
}