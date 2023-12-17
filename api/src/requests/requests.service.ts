import {Injectable, NotFoundException} from "@nestjs/common";
import {RequestsCreateDto} from "./dto/requests-create.dto";
import {MinioService} from "../minio/minio.service";
import {BufferedFile} from "../minio/dto/file.dto";
import {RequestEntity} from "./entities/request.entity";
import {Between, FindOptionsWhere, Repository} from "typeorm";
import {InjectRepository} from "@nestjs/typeorm";
import {RequestsDto} from "./dto/requests.dto";
import {RabbitmqService} from "../rabbitmq/rabbitmq.service";
import {RequestsStatusInterface} from "./interfaces/requests.status.interface";
import {RabbitmqRequestsDto} from "../rabbitmq/dto/rabbitmq-requests.dto";
import {RequestsDumpParamsDto} from "./dto/requests-dump-params.dto";
import {DateTime} from "luxon";
import {RequestsUpdateDto} from "./dto/requests-update.dto";

@Injectable()
export class RequestsService {
  constructor(
    @InjectRepository(RequestEntity) private requestEntityRepository: Repository<RequestEntity>,
    private minioClientService: MinioService,
    private rabbitmqService: RabbitmqService,
  ) {
  }

  private transformEntityToDto(entity: RequestEntity): RequestsDto {
    return new RequestsDto(entity)
  }

  async find(): Promise<RequestsDto[]> {
    const items: RequestsDto[] = []
    const entities = await this.requestEntityRepository.find()

    for (let i = 0; i < entities.length; i++) {
      items.push(this.transformEntityToDto(entities[i]))
    }

    return items
  }

  async download({date_from, date_to, rating}: RequestsDumpParamsDto) {
    const items: RequestsDto[] = []
    const where: FindOptionsWhere<RequestEntity> = {}

    console.log(date_from, date_to, rating)

    if (rating) {
      where.rating = rating
    }

    const entities = await this.requestEntityRepository.find({
      where: {
        created_at: Between(date_from, date_to),
        ...where
      }
    })

    for (let i = 0; i < entities.length; i++) {
      items.push(this.transformEntityToDto(entities[i]))
    }

    return items
  }

  async findById(id: number): Promise<RequestsDto> {
    const entity = await this.requestEntityRepository.findOne({where: {id: id}})
    if (!entity) throw new NotFoundException()

    return this.transformEntityToDto(entity)
  }

  async updateRatingById(id: number, dto: RequestsUpdateDto): Promise<RequestsDto> {
    const entity = await this.requestEntityRepository.findOne({where: {id: id}})
    if (!entity) throw new NotFoundException()

    await this.requestEntityRepository.update({
      id: entity.id,
    }, {
      rating: dto.rating
    })

    return this.transformEntityToDto({...entity, rating: dto.rating})
  }

  async updateStatus(data: RabbitmqRequestsDto) {
    await this.requestEntityRepository.update({id: data.id}, {
      status: RequestsStatusInterface.success,
      response: {
        text: data.data.text.substring(0, 2500),
        confidence: data.data.confidence
      }
    })
  }

  async create(body: RequestsCreateDto, files?: Array<BufferedFile>) {
    const uploadedImages: string[] = [];

    if (files) {
      for (let i = 0; i < files.length; i++) {
        const image = await this.minioClientService.upload(files[i]);
        uploadedImages.push(image);
      }
    }

    const requestObject = this.requestEntityRepository.create({
      data: body,
      original_images: uploadedImages,
    })

    const request = await this.requestEntityRepository.save(requestObject)
    const response = this.transformEntityToDto(request)

    await this.rabbitmqService.sendMessage(response)

    return response
  }
}
