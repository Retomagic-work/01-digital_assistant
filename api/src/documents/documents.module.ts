import { Module } from '@nestjs/common';
import { DocumentsService } from './documents.service';
import { DocumentsController } from './documents.controller';
import {TypeOrmModule} from "@nestjs/typeorm";
import {DocumentsEntity} from "./entities/documents.entity";
import {MinioModule} from "../minio/minio.module";
import {RabbitmqModule} from "../rabbitmq/rabbitmq.module";

@Module({
  imports: [
    TypeOrmModule.forFeature([DocumentsEntity]),
    MinioModule,
    RabbitmqModule
  ],
  providers: [DocumentsService],
  controllers: [DocumentsController],
  exports: [DocumentsService]
})
export class DocumentsModule {}
