import {Transform} from "class-transformer";

export class DocumentsDto {
  id: number
  description: string

  @Transform(({value}) => (`${process.env.SERVER_ADDRESS}files/documents/${value}`))
  link: string

  created_at: Date

  constructor(partial: Partial<DocumentsDto>) {
    Object.assign(this, partial);
  }
}