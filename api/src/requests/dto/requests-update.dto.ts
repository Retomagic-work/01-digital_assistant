import {IsOptional} from "class-validator";

export class RequestsUpdateDto {
    @IsOptional()
    rating?: number
}