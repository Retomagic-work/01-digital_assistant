import {Module} from "@nestjs/common";
import {RequestsService} from "./requests.service";
import {RequestsController} from "./requests.controller";
import {MinioModule} from "../minio/minio.module";
import {TypeOrmModule} from "@nestjs/typeorm";
import {RequestEntity} from "./entities/request.entity";
import {RabbitmqModule} from "../rabbitmq/rabbitmq.module";

@Module({
    imports: [
        TypeOrmModule.forFeature([RequestEntity]),
        RabbitmqModule,
        MinioModule
    ],
    controllers: [RequestsController],
    providers: [RequestsService],
    exports: [RequestsService],
})
export class RequestsModule {
}
