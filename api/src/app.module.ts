import {Module} from "@nestjs/common";
import {ConfigModule} from "@nestjs/config";
import {RequestsModule} from "./requests/requests.module";
import {TypeOrmModule} from "@nestjs/typeorm";
import {MinioModule} from './minio/minio.module';
import {RabbitmqModule} from "./rabbitmq/rabbitmq.module";
import {UsersModule} from "./users/users.module";
import {AuthModule} from "./auth/auth.module";
import {DocumentsModule} from "./documents/documents.module";

@Module({
  imports: [
    ConfigModule.forRoot({
      envFilePath: [".env"]
    }),
    TypeOrmModule.forRoot({
      type: "postgres",
      port: 5432,
      host: process.env.POSTGRES_HOST,
      username: process.env.POSTGRES_USER,
      password: process.env.POSTGRES_PASSWORD,
      database: process.env.POSTGRES_DB,
      synchronize: true,
      autoLoadEntities: true
    }),
    RabbitmqModule,
    MinioModule,
    RequestsModule,
    UsersModule,
    AuthModule,
    DocumentsModule
  ],
  controllers: [],
  providers: []
})
export class AppModule {
}
