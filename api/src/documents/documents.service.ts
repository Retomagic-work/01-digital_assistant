import {Injectable, NotFoundException} from '@nestjs/common';
import {InjectRepository} from "@nestjs/typeorm";
import {Repository} from "typeorm";
import {MinioService} from "../minio/minio.service";
import {DocumentsEntity} from "./entities/documents.entity";
import {DocumentsDto} from "./dto/documents.dto";
import {DocumentsUpdateDto} from "./dto/documents-update.dto";
import {BufferedFile} from "../minio/dto/file.dto";
import {RabbitmqService} from "../rabbitmq/rabbitmq.service";

@Injectable()
export class DocumentsService {
  constructor(
    @InjectRepository(DocumentsEntity) private documentsEntityRepository: Repository<DocumentsEntity>,
    private minioClientService: MinioService,
    private rabbitmqService: RabbitmqService,
  ) {
  }

  private transformEntityToDto(entity: DocumentsEntity): DocumentsDto {
    return new DocumentsDto(entity)
  }

  async find(): Promise<DocumentsDto[]> {
    const items: DocumentsDto[] = []
    const entities = await this.documentsEntityRepository.find()

    for (let i = 0; i < entities.length; i++) {
      items.push(this.transformEntityToDto(entities[i]))
    }

    return items
  }

  async update(id: number, dto: DocumentsUpdateDto): Promise<DocumentsDto> {
    const entity = await this.documentsEntityRepository.findOne({where: {id: id}})
    if (!entity) throw new NotFoundException()

    await this.documentsEntityRepository.update({
      id: entity.id,
    }, {
      description: dto.description
    })

    return this.transformEntityToDto({...entity, description: dto.description})
  }

  async create(files?: Array<BufferedFile>): Promise<DocumentsDto[]> {
    const documents: DocumentsDto[] = [];

    if (files) {
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        const link = await this.minioClientService.upload(file, 'documents');

        console.log(file)

        const document = await this.documentsEntityRepository.save({
          description: file.originalname,
          link: link
        })

        const _document = this.transformEntityToDto(document)

        await this.rabbitmqService.sendMessageUpload(_document)
        documents.push(_document);
      }
    }

    return documents
  }
}
