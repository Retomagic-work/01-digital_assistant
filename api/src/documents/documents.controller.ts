import {
  Body,
  ClassSerializerInterceptor,
  Controller,
  Get, Param,
  Post, Put,
  UploadedFiles,
  UseGuards,
  UseInterceptors
} from '@nestjs/common';
import {ApiOperation, ApiTags} from "@nestjs/swagger";
import {Throttle} from "@nestjs/throttler";
import {JwtAuthGuard} from "../auth/guards/jwt-auth.guard";
import {FilesInterceptor} from "@nestjs/platform-express";
import {RequestsCreateDto} from "../requests/dto/requests-create.dto";
import {BufferedFile} from "../minio/dto/file.dto";
import {PdfFileMimeTypeValidationPipe} from "../common/pipes/PdfFileMimeTypeValidationPipe";
import {DocumentsUpdateDto} from "./dto/documents-update.dto";
import {DocumentsDto} from "./dto/documents.dto";
import {DocumentsService} from "./documents.service";

@ApiTags("Документы")
@Controller('documents')
export class DocumentsController {
  constructor(private readonly documentsService: DocumentsService) {
  }

  @ApiOperation({summary: "Получение документов (Only auth)"})
  @Get()
  @Throttle(60, 60)
  @UseGuards(JwtAuthGuard)
  @UseInterceptors(ClassSerializerInterceptor)
  async find(): Promise<DocumentsDto[]> {
    try {
      return await this.documentsService.find();
    } catch (e) {
      throw e;
    }
  }

  @ApiOperation({summary: "Обновление ифнормации документа (Only auth)"})
  @Put(':id')
  @Throttle(4, 60)
  @UseGuards(JwtAuthGuard)
  @UseInterceptors(ClassSerializerInterceptor)
  async update(@Body() body: DocumentsUpdateDto, @Param('id') id: number): Promise<DocumentsDto> {
    try {
      return await this.documentsService.update(id, body);
    } catch (e) {
      throw e;
    }
  }

  @ApiOperation({summary: "Создание документа (Only auth)"})
  @Post()
  @Throttle(4, 60)
  @UseInterceptors(FilesInterceptor("files", 1000), ClassSerializerInterceptor)
  @UseGuards(JwtAuthGuard)
  async create(@Body() body: RequestsCreateDto, @UploadedFiles(PdfFileMimeTypeValidationPipe) files: Array<BufferedFile>): Promise<DocumentsDto[]> {
    try {
      return await this.documentsService.create(files);
    } catch (e) {
      throw e;
    }
  }
}
