import {
  Body, ClassSerializerInterceptor,
  Controller, Get, Param,
  Post, Put, Query,
  UploadedFiles, UseGuards,
  UseInterceptors
} from "@nestjs/common";
import {RequestsService} from "./requests.service";
import {Throttle} from "@nestjs/throttler";
import {FilesMimeTypeValidationPipe} from "../common/pipes/FileMimeTypeValidationPipe";
import {FilesInterceptor} from "@nestjs/platform-express";
import {RequestsCreateDto} from "./dto/requests-create.dto";
import {BufferedFile} from "../minio/dto/file.dto";
import {RequestsDto} from "./dto/requests.dto";
import {JwtAuthGuard} from "../auth/guards/jwt-auth.guard";
import {ApiOperation, ApiTags} from "@nestjs/swagger";
import {RequestsDumpParamsDto} from "./dto/requests-dump-params.dto";
import {RequestsUpdateDto} from "./dto/requests-update.dto";

@ApiTags("Запросы")
@Controller("requests")
export class RequestsController {
  constructor(private readonly requestsService: RequestsService) {
  }

  @ApiOperation({ summary: "Получение запроса по ID" })
  @Get(':id')
  @Throttle(6000, 60)
  @UseInterceptors(ClassSerializerInterceptor)
  async getOne(@Param('id') id: number): Promise<RequestsDto> {
    try {
      return await this.requestsService.findById(id);
    } catch (e) {
      throw e;
    }
  }

  @ApiOperation({ summary: "Выставить рейтинг запросу (Only auth)" })
  @Put(':id')
  @Throttle(60, 60)
  @UseInterceptors(ClassSerializerInterceptor)
  @UseGuards(JwtAuthGuard)
  async update(@Param('id') id: number, @Body() dto: RequestsUpdateDto): Promise<RequestsDto> {
    try {
      return await this.requestsService.updateRatingById(id, dto);
    } catch (e) {
      throw e;
    }
  }

  @ApiOperation({ summary: "Получение запросов (Only auth)" })
  @Get()
  @Throttle(60, 60)
  @UseInterceptors(ClassSerializerInterceptor)
  @UseGuards(JwtAuthGuard)
  async get(): Promise<RequestsDto[]> {
    try {
      return await this.requestsService.find();
    } catch (e) {
      throw e;
    }
  }

  @ApiOperation({ summary: "Выгрузить(Скачать) список запросов. (Only auth)." })
  @Get('download/dump')
  @Throttle(60, 60)
  async download(@Query() query: RequestsDumpParamsDto) {
    try {
      return this.requestsService.download(query);
    } catch (e) {
      throw e;
    }
  }

  @ApiOperation({ summary: "Создание запроса" })
  @Post()
  @Throttle(4, 60)
  @UseInterceptors(FilesInterceptor("files", 10), ClassSerializerInterceptor)
  async create(@Body() body: RequestsCreateDto, @UploadedFiles(FilesMimeTypeValidationPipe) files: Array<BufferedFile>): Promise<RequestsDto> {
    try {
      return await this.requestsService.create(body, files);
    } catch (e) {
      throw e;
    }
  }
}