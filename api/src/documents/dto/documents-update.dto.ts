import {IsOptional} from "class-validator";

export class DocumentsUpdateDto {
    @IsOptional()
    description: string
}