import {IsNotEmpty, IsOptional} from "class-validator";
import {Transform} from "class-transformer";
import {DateTime} from "luxon";

export class RequestsDumpParamsDto {
  @IsNotEmpty()
  @Transform(({value}) => DateTime.fromISO(value).toJSDate())
  date_from: Date;

  @IsNotEmpty()
  @Transform(({value}) => DateTime.fromISO(value).toJSDate())
  date_to: Date;

  @IsOptional()
  rating?: number;
}