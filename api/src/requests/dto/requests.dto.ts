import {RequestsStatusInterface} from "../interfaces/requests.status.interface";
import {RequestsDataDto} from "./requests-data.dto";
import {Transform} from "class-transformer";

export class RequestsDto {
  id: number

  rating: number

  status: RequestsStatusInterface

  @Transform(({value}) => value.map((filename: string) => (`${process.env.SERVER_ADDRESS}files/requests/${filename}`)))
  original_images: string[]

  @Transform(({value}) => value.map((filename: string) => (`${process.env.SERVER_ADDRESS}files/responses/${filename}`)))
  finished_images: string[]

  data: RequestsDataDto

  created_at: Date
  updated_at: Date

  constructor(partial: Partial<RequestsDto>) {
    Object.assign(this, partial)
  }
}