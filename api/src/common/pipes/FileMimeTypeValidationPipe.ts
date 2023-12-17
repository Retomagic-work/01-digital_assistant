import { PipeTransform, Injectable, ArgumentMetadata, HttpException, HttpStatus } from "@nestjs/common";

@Injectable()
export class FilesMimeTypeValidationPipe implements PipeTransform {
  transform(values: any[], metadata: ArgumentMetadata) {
    if (values) {

      let errors: any[] = [];
      for (let i = 0; i < values.length; i++) {
        const file = values[i];

        if (
          file.mimetype !== "image/png" &&
          file.mimetype !== "image/jpeg" &&
          file.mimetype !== "image/heic"
        ) {
          errors.push({ text: "Unauthorized file format loaded", file: file });
        }

        if ((file.size / 1000000) > 65) {
          errors.push({ text: "File size too large > 65mb", file: file });
        }
      }

      if (errors.length === 0) {
        return values;
      } else {
        throw new HttpException({
          status: HttpStatus.UNSUPPORTED_MEDIA_TYPE,
          error: errors.map(er => ({
            text: er.text, file: {
              originalname: er.file.originalname,
              mimetype: er.file.mimetype,
              size: er.file.size
            }
          }))
        }, HttpStatus.UNSUPPORTED_MEDIA_TYPE);
      }
    }
  }
}